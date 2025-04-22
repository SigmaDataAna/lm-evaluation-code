#!/bin/bash

exoprt sas_key="?sv=2023-01-03&st=2025-04-22T06%3A25%3A08Z&se=2025-04-29T06%3A24%3A00Z&skoid=93dcab78-2e9c-4cca-8417-3c59080fb09d&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-04-22T06%3A25%3A08Z&ske=2025-04-29T06%3A24%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=dyI9XaeSwu5Jp3JNKOKIL1lDJ%2Fh%2F2VaWDOsImmOO2pY%3D"

azcopy copy --recursive "https://hptrainingfrancecentral.blob.core.windows.net/pretraining/checkpoints/OpenPAI-Pretrain-17BA3B-RoPE-HQ-0405/hf_iter_206000/${sas_key}" ..