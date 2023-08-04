import numpy as np
import mindspore as ms

from transformers.activations import NewGELUActivation
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerNorm


class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


if __name__ == '__main__':
    # x = np.load("./debug_t5/random_input.npy")
    # x = ms.Tensor(x)
    # layer = NewGELUActivation()
    # y = layer(x)
    # y = y.numpy()
    # print(y)
    # np.save("./debug_t5/random_output.npy", y)

    # relative_position = np.load("./debug_t5/encoder.block.0.layer.0.SelfAttention.rp.npy")
    # relative_position = ms.Tensor(relative_position)
    # layer = T5Attention._relative_position_bucket
    # relative_position_bucket = layer(relative_position, True, 32, 128)
    # relative_position_bucket = relative_position_bucket.numpy()
    # print(relative_position_bucket)

    hidden_states = ms.ops.randn((2, 77, 4096))
    mask = ms.Tensor(np.load("./debug_t5/mask.npy"), dtype=ms.float32)
    layer = T5Attention(
        ObjDict(is_decoder=False, relative_attention_num_buckets=32, relative_attention_max_distance=128,
                d_model=4096, d_kv=64, num_heads=64, dropout_rate=0.1),
        has_relative_attention_bias=True
    )
    for name, module in layer.cells_and_names(name_prefix="SA"):
        module.debug_name = name
    y = layer(hidden_states, mask)
    y = y.numpy()
    print(y)
