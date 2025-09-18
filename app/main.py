from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Invoice & Receipt Parser API")

# =========================
# Model & Processor Loading
# =========================
MODEL_ID = "mychen76/invoice-and-receipts_donut_v1"

# Load processor
processor = DonutProcessor.from_pretrained(MODEL_ID, use_fast=True)

# Force full CPU load (no meta tensors, no safetensors memory map)
model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_ID,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
    use_safetensors=False,   # 🚨 important: force standard torch load
    dtype=torch.float32
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# =========================
# API Endpoint
# =========================
@app.post("/parse-receipt/")
async def parse_receipt(file: UploadFile = File(...)):
    # Read file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Prompt
    task_prompt = "<s_receipt>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    # Generate
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode
    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    result = processor.token2json(seq)

    return JSONResponse(content=result)
