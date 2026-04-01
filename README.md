Vibevoice code for exllama v3 about 4x as fast as transformers with fp16.

To install and infer.

1. CD to VibeVoice directory.
2. Run "pip install -e ."
3. Run "pip uninstall torch torchvision torchaudio"
3. Run "pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
4. Run "pip install gradio"
5. Run "pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.27/exllamav3-0.0.27+cu128.torch2.8.0-cp312-cp312-linux_x86_64.whl"
6. Git clone "https://huggingface.co/dalazymodder/vibevoice_asr_exllama_q8"
6. Run "vibevoice_asr_exl3_inference.py or vibevoice_asr_gradio.py"


NOTE:
You may need to adjust pytorch version and exllamav3 versions to match your python version.


To split the llm for smaller quants.

1. Git clone microsoft/VibeVoice-ASR
2. Run python split_vibevoice_asr.py --model_dir VibeVoice-ASR
3. Git clone https://github.com/turboderp-org/exllamav3
4. CD to exllamav3
5. Run pip install -r requirements.txt
6. Copy the tokenizers from Qwen/Qwen2.5-7B-Instruct to the split/llm directory.
7. CD back up a level and run command "python exllamav3/convert.py -i split/llm -o split/vibex -w tmp -b 8"
Adjust -b to quant level.
