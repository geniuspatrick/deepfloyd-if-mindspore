from deepfloyd_if.modules import IFStageI, IFStageII  # , StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream


t5 = T5Embedder()
if_I = IFStageI('IF-I-M-v1.0')
if_II = IFStageII('IF-II-M-v1.0')
# if_III = StableStageIII('stable-diffusion-x4-upscaler')
for name, module in t5.model.cells_and_names():
    module.debug_name = name

prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'
count = 4

t5_embs = t5.get_text_embeddings([prompt] * 2)
print(t5_embs)
import numpy as np
np.save("text_encoder_embs.npy", t5_embs.numpy())
exit(0)

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=None,
    prompt=[prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
    if_III_kwargs={
        "guidance_scale": 9.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
)

if_II.show(result["II"])

if_III.show(result['III'], size=14)
