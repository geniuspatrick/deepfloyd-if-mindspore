from deepfloyd_if.modules import IFStageI, IFStageII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream


t5 = T5Embedder()
if_I = IFStageI('IF-I-XL-v1.0')
if_II = IFStageII('IF-II-L-v1.0')
# if_III = StableStageIII('stable-diffusion-x4-upscaler')


prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'
count = 4

result = dream(
    t5=t5, if_I=if_I, if_II=if_II,  # if_III=if_III,
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

if_III.show(result['III'], size=14)
