import argparse
import tensorrt as trt
import os

def build_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # 显存限制，按需调整（单位字节，这里设 8GB）
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    # 解析 ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[ONNX Parse Error] {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")
    print(f"ONNX parsed successfully: {onnx_path}")

    # 创建 optimization profile
    profile = builder.create_optimization_profile()

    # ---------- 从截图读取的 min / opt / max ----------
    # x      : (batch, out_channels, seq_len)
    profile.set_shape("x",    (2, 80,  4),  (2, 80, 193),  (2, 80, 6800))
    # mask   : (batch, 1, seq_len)
    profile.set_shape("mask", (2,  1,  4),  (2,  1, 193),  (2,  1, 6800))
    # mu     : (batch, out_channels, seq_len)
    profile.set_shape("mu",   (2, 80,  4),  (2, 80, 193),  (2, 80, 6800))
    # t      : (batch,)
    profile.set_shape("t",    (2,),          (2,),           (2,))
    # spks   : (batch, out_channels)
    profile.set_shape("spks", (2, 80),       (2, 80),        (2, 80))
    # cond   : (batch, out_channels, seq_len)
    profile.set_shape("cond", (2, 80,  4),  (2, 80, 193),  (2, 80, 6800))
    # -------------------------------------------------

    config.add_optimization_profile(profile)

    # 序列化构建引擎
    print("Building TRT engine, this may take a while...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TRT engine")

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"TRT engine saved to: {engine_path}")
    return engine_path


def verify_engine(engine_path):
    """简单验证引擎可以正常加载"""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    print(f"Engine loaded successfully, num_io_tensors: {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        mode = engine.get_tensor_mode(name)
        print(f"  {'INPUT' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'}: {name} {shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", default="models/Decoder_trt/flow.decoder.estimator.fp32.onnx")
    parser.add_argument("--engine_path", default="models/Decoder_trt/flow.decoder.estimator.fp32.a800.plan")
    args = parser.parse_args()
    onnx_path = args.onnx_path
    engine_path = args.engine_path

    build_engine(onnx_path, engine_path, fp16=False)
    verify_engine(engine_path)