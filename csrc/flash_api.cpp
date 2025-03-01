// Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/fast_math.h>

#include "flash_mla.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

/**
 * get_mla_metadata() 用于获取 MLA Kernel的元数据
 * 用于计算张量的块分配元数据， 便于 MLA 中高效地利用 Hopper 资源
 * 主要基于 seqlens_k、num_heads_per_head_k 和 num_heads_k 计算所需的元数据
 */
std::vector<at::Tensor>
get_mla_metadata(
    at::Tensor &seqlens_k,
    const int num_heads_per_head_k,
    const int num_heads_k
) {
    // This should match the logic in the MLA kernel.
    static constexpr int block_size_m = 64;
    static constexpr int block_size_n = 64;
    static constexpr int fixed_overhead_num_blocks = 5;

    // 检查输入张量是否在CUDA设备上
    CHECK_DEVICE(seqlens_k);
    TORCH_CHECK(seqlens_k.is_contiguous());
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);

    int batch_size = seqlens_k.size(0);
    int *seqlens_k_ptr = seqlens_k.data_ptr<int>();
    auto options = seqlens_k.options();

    // 获取当前CUDA设备的属性
    auto dprops = at::cuda::getCurrentDeviceProperties();
    int sm_count = dprops->multiProcessorCount;
    int num_sm_parts = sm_count / num_heads_k / cutlass::ceil_div(num_heads_per_head_k, block_size_m);

    // 创建用于存储元数据的张量
    auto tile_scheduler_metadata = torch::empty({num_sm_parts, TileSchedulerMetaDataSize}, options);
    auto num_splits = torch::empty({batch_size + 1}, options);
    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int *num_splits_ptr = num_splits.data_ptr<int>();

    // 设置CUDA设备保护，确保在正确的设备上执行
    at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // 设置元数据参数
    Mla_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;

    // 调用CUDA内核函数获取元数据
    get_mla_metadata_func(params, stream);

    return {tile_scheduler_metadata, num_splits};
}

/**
 * Flash-MLA 的前向传播函数，用于高效计算多头注意力机制 MLA
 */
std::vector<at::Tensor>
mha_fwd_kvcache_mla(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
    c10::optional<const at::Tensor> &vcache_,    // num_blocks x page_block_size x num_heads_k x head_size_v
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits                 // batch_size + 1
) {
    // 检查CUDA设备是否为SM90架构，Hopper 架构
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90);

    // 如果未提供vcache，则使用kcache作为vcache
    at::Tensor vcache = vcache_.has_value() ? vcache_.value() : kcache;

    // 检查输入张量的数据类型和形状
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kBFloat16);
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    CHECK_DEVICE(block_table);
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");

    // 获取输入张量的形状信息
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_ori = sizes[2];
    const int head_size = sizes[3];
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(num_heads_ori % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // 如果seqlen_q为1，则不需要 Casual Mask
    if (seqlen_q_ori == 1) { is_causal = false; }

    // 重新调整 q 的 shape 以适应 Multi-Head Attention
    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;
    q = q.view({batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size}).transpose(2, 3)
            .reshape({batch_size, seqlen_q, num_heads, head_size});

    int head_size_k = head_size;
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    if (vcache_.has_value()) { CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_v); }
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);

    // 检查seqlens_k的形状和数据类型
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    CHECK_DEVICE(seqlens_k);
    CHECK_CONTIGUOUS(seqlens_k);
    CHECK_SHAPE(seqlens_k, batch_size);

    // 设置CUDA设备保护
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // 创建输出张量
    auto opts = q.options();
    at::Tensor out = torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts);
    at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    // 设置前向传播参数
    Flash_fwd_mla_params params = {};
    // Set the sizes.
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.cu_seqlens_k = seqlens_k.data_ptr<int>();
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = ngroups;
    params.is_causal = is_causal;
    params.d = head_size;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);

    // 设置指针和步幅
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.v_ptr = vcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.v_batch_stride = vcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kcache.stride(-3);
    params.v_row_stride = vcache.stride(-3);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kcache.stride(-2);
    params.v_head_stride = vcache.stride(-2);
    params.o_head_stride = out.stride(-2);

    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;

    // 检查tile_scheduler_metadata的形状和数据类型
    TORCH_CHECK(tile_scheduler_metadata.dtype() == torch::kInt32, "tile_scheduler_metadata must have dtype int32");
    TORCH_CHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize);
    CHECK_DEVICE(tile_scheduler_metadata);
    CHECK_CONTIGUOUS(tile_scheduler_metadata);
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_sm_parts = tile_scheduler_metadata.size(0);
    // 检查num_splits的形状和数据类型
    TORCH_CHECK(num_splits.dtype() == torch::kInt32, "num_splits must have dtype int32");
    CHECK_DEVICE(num_splits);
    CHECK_CONTIGUOUS(num_splits);
    params.num_splits_ptr = num_splits.data_ptr<int>();

    // 创建用于累积的中间张量
    at::Tensor softmax_lse_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor out_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q, head_size_v}, opts.dtype(at::kFloat));
    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();

    // 获取当前CUDA Stream并运行前向传播的 MLA Kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(head_size == 576);
    run_mha_fwd_splitkv_mla<cutlass::bfloat16_t, 576>(params, stream);

    // 调整输出张量的形状
    out = out.view({batch_size, seqlen_q_ori, ngroups, num_heads_k, head_size_v}).transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_ori, head_size_v});
    softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, ngroups}).transpose(2, 3)
            .reshape({batch_size, num_heads_ori, seqlen_q_ori});

    return {out, softmax_lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("get_mla_metadata", &get_mla_metadata);
    m.def("fwd_kvcache_mla", &mha_fwd_kvcache_mla);
}
