import numpy as np


if __name__ == '__main__':
    root_ms = "/Users/wyf/DevPy/deepfloyd-if-mindspore/debug_t5"
    root_pt = "/Users/wyf/DevPy/deepfloyd-if/debug_t5"
    feat_name = "text_encoder_embs.npy"
    feat_ms = np.load(f"{root_ms}/{feat_name}")
    feat_pt = np.load(f"{root_pt}/{feat_name}")
    feat_diff = feat_pt - feat_ms
    feat_diff_abs = np.abs(feat_diff)
    print(np.max(feat_diff_abs))
    print(np.allclose(feat_ms, feat_pt))

    # import matplotlib.pyplot as plt
    # diffs = []
    # for b in range(24):
    #     feat_name = f"encoder.block.{b}.npy"
    #     feat_ms = np.load(f"{root_ms}/{feat_name}")
    #     feat_pt = np.load(f"{root_pt}/{feat_name}")
    #     feat_diff = feat_pt - feat_ms
    #     feat_diff_abs = np.abs(feat_diff)
    #     diffs.append(feat_diff_abs.max())
    # print(diffs)
    # plt.plot(diffs)
    # plt.show()

    import matplotlib.pyplot as plt
    diffs = []
    for b in range(24):
        diff = []
        for feat_name in [
            f"encoder.block.{b}.layer.0.layer_norm.npy",
            f"encoder.block.{b}.layer.0.SelfAttention.q.npy",
            f"encoder.block.{b}.layer.0.SelfAttention.k.npy",
            f"encoder.block.{b}.layer.0.SelfAttention.v.npy",
            f"encoder.block.{b}.layer.0.SelfAttention.s.npy",
            # f"encoder.block.{b}.layer.0.SelfAttention.pb.npy",
            f"encoder.block.{b}.layer.0.SelfAttention.aw.npy",
            f"encoder.block.{b}.layer.0.SelfAttention.ao.npy",
            f"encoder.block.{b}.layer.0.SelfAttention.npy",
            f"encoder.block.{b}.layer.0.npy",
            f"encoder.block.{b}.layer.1.npy",
            f"encoder.block.{b}.npy",
        ]:
            feat_ms = np.load(f"{root_ms}/{feat_name}")
            feat_pt = np.load(f"{root_pt}/{feat_name}")
            feat_diff = feat_pt - feat_ms
            feat_diff_abs = np.abs(feat_diff)
            diff.append(feat_diff_abs.max())
        print([f"{d:e}" for d in diff])
        diffs.extend(diff)
    print([f"{d:<12}" for d in [
        "LSA.LN",
        "LSA.SA.q",
        "LSA.SA.k",
        "LSA.SA.v",
        "LSA.SA.s",
        "LSA.SA.aw",
        "LSA.SA.ao",
        "LSA.SA",
        "LSA",
        "FF",
    ]])
    plt.plot(diffs)
    plt.savefig("debug_t5/diff.jpg")
    plt.show()
