
import json
import random
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = ROOT / "qwen_lora_adapter"
REPORT_PATH = ROOT / "evaluation_qwen.json"

SEED = 42
EPOCHS = 5
ACCUMULATION = 8
MAX_LENGTH = 256
LEARNING_RATE = 0.00015
PATIENCE = 2

random.seed(SEED)
torch.manual_seed(SEED)

CATALOG = [
    ("Смартфон", ["Samsung Galaxy A55", "Xiaomi Redmi Note 13", "Realme 12", "Honor 90"], ["128 ГБ", "256 ГБ", "512 ГБ"]),
    ("Ноутбук", ["Lenovo IdeaPad", "ASUS VivoBook", "Acer Aspire", "HP Pavilion"], ["8 ГБ ОЗУ", "16 ГБ ОЗУ", "32 ГБ ОЗУ"]),
    ("Наушники", ["Sony WH-CH720N", "JBL Tune 770NC", "Anker Soundcore Q30", "Edifier W820NB"], ["Bluetooth 5.0", "Bluetooth 5.2", "Bluetooth 5.3"]),
    ("Монитор", ["LG UltraGear", "Samsung Odyssey", "AOC Gaming", "Dell UltraSharp"], ["24 дюйма", "27 дюймов", "32 дюйма"]),
    ("Планшет", ["Apple iPad", "Samsung Galaxy Tab", "Xiaomi Pad", "Lenovo Tab"], ["64 ГБ", "128 ГБ", "256 ГБ"]),
    ("Колонка", ["JBL Flip", "Sony SRS", "Anker Soundcore", "Marshall Emberton"], ["Bluetooth 5.0", "Bluetooth 5.1", "Bluetooth 5.3"]),
    ("Мышь", ["Logitech MX Master", "Razer Basilisk", "A4Tech Fstyler", "Microsoft Bluetooth Mouse"], ["Bluetooth", "USB-приёмник", "беспроводное подключение"]),
    ("Внешний SSD", ["Samsung T7", "Kingston XS1000", "SanDisk Extreme", "ADATA SE880"], ["500 ГБ", "1 ТБ", "2 ТБ"])
]


def build_data():
    train = []
    validation = []
    test = []

    for category, brands, specs in CATALOG:
        for index, brand in enumerate(brands):
            for spec in specs:
                source = f"{category} {brand}, {spec}"
                target = f"{category} {brand} с характеристикой {spec}."

                row = {
                    "товар": source,
                    "аннотация": target,
                    "бренд": brand,
                    "характеристика": spec
                }

                if index < 2:
                    train.append(row)
                elif index == 2:
                    validation.append(row)
                else:
                    test.append(row)

    for part in (train, validation, test):
        random.shuffle(part)

    return train, validation, test


def messages(source, target=None):
    result = [
        {
            "role": "system",
            "content": (
                "Ты создаёшь короткие аннотации товаров на русском языке. "
                "Сохраняй название, модель и характеристики точно. "
                "Не добавляй неизвестные свойства."
            )
        },
        {
            "role": "user",
            "content": source
        }
    ]

    if target is not None:
        result.append({
            "role": "assistant",
            "content": target
        })

    return result



def encode(row, tokenizer):
    prompt_ids = tokenizer.apply_chat_template(
        messages(row["товар"]),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None
    )

    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids

    answer_ids = tokenizer(
        row["аннотация"],
        add_special_tokens=False
    )["input_ids"]

    end_ids = tokenizer(
        "<|im_end|>",
        add_special_tokens=False
    )["input_ids"]

    input_ids = (
        list(prompt_ids)
        + list(answer_ids)
        + list(end_ids)
    )

    labels = (
        [-100] * len(prompt_ids)
        + list(answer_ids)
        + list(end_ids)
    )

    if len(input_ids) > MAX_LENGTH:
        raise ValueError(
            "Пример слишком длинный для обучения"
        )

    if len(answer_ids) == 0:
        raise ValueError(
            "В примере отсутствуют токены ответа"
        )

    return {
        "input_ids": torch.tensor(
            [input_ids],
            dtype=torch.long,
            device="cuda"
        ),
        "attention_mask": torch.ones(
            (1, len(input_ids)),
            dtype=torch.long,
            device="cuda"
        ),
        "labels": torch.tensor(
            [labels],
            dtype=torch.long,
            device="cuda"
        )
    }



def validation_loss(model, rows, tokenizer):
    model.eval()
    total = 0.0

    with torch.no_grad():
        for row in rows:
            batch = encode(row, tokenizer)
            total += model(**batch).loss.item()

    return total / len(rows)



def generate(model, tokenizer, source):
    model.eval()

    prompt = tokenizer.apply_chat_template(
        messages(source),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    if hasattr(prompt, "input_ids"):
        prompt = prompt.input_ids

    prompt = prompt.to("cuda")

    with torch.no_grad():
        output = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    generated = output[0, prompt.shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip().split("\n")[0]



def evaluate(model, tokenizer, rows):
    results = []

    for row in rows:
        prediction = generate(
            model,
            tokenizer,
            row["товар"]
        )

        results.append({
            "товар": row["товар"],
            "эталон": row["аннотация"],
            "предсказание": prediction,
            "бренд_сохранён": (
                row["бренд"].casefold()
                in prediction.casefold()
            ),
            "характеристика_сохранена": (
                row["характеристика"].casefold()
                in prediction.casefold()
            )
        })

    return results


def score(rows):
    count = len(rows)

    return {
        "точное_совпадение": round(
            sum(
                row["предсказание"].strip()
                == row["эталон"]
                for row in rows
            ) / count,
            4
        ),
        "сохранение_бренда": round(
            sum(
                row["бренд_сохранён"]
                for row in rows
            ) / count,
            4
        ),
        "сохранение_характеристики": round(
            sum(
                row["характеристика_сохранена"]
                for row in rows
            ) / count,
            4
        )
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Видеокарта CUDA недоступна"
        )

    print(
        f"Устройство: {torch.cuda.get_device_name(0)}"
    )

    train, validation, test = build_data()

    print(
        f"Обучение: {len(train)} | "
        f"Проверка: {len(validation)} | "
        f"Тест: {len(test)}"
    )

    print("Загрузка токенизатора...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    tokenizer.pad_token = tokenizer.eos_token

    print("Загрузка языковой модели...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cuda")

    model.config.use_cache = False

    print("Оценка исходной модели...")

    baseline = evaluate(
        model,
        tokenizer,
        test
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE
    )

    scaler = torch.amp.GradScaler("cuda")

    best_loss = float("inf")
    patience = 0
    history = []

    print("Начало обучения LoRA...")

    for epoch in range(EPOCHS):
        random.shuffle(train)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total = 0.0

        for index, row in enumerate(train):
            batch = encode(row, tokenizer)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                loss = model(**batch).loss

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Обнаружена некорректная функция потерь"
                )

            scaler.scale(
                loss / ACCUMULATION
            ).backward()

            total += loss.item()

            if (
                (index + 1) % ACCUMULATION == 0
                or index + 1 == len(train)
            ):
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        train_loss = total / len(train)

        validation_value = validation_loss(
            model,
            validation,
            tokenizer
        )

        history.append({
            "эпоха": epoch + 1,
            "обучение": round(train_loss, 5),
            "проверка": round(validation_value, 5)
        })

        print(
            f"Эпоха {epoch + 1}/{EPOCHS} | "
            f"Обучение: {train_loss:.4f} | "
            f"Проверка: {validation_value:.4f}"
        )

        if validation_value < best_loss - 0.0001:
            best_loss = validation_value
            patience = 0

            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

            print("Лучший адаптер сохранён")

        else:
            patience += 1

            if patience >= PATIENCE:
                print("Ранняя остановка обучения")
                break

    print("Загрузка лучшего адаптера...")

    adapter_path = OUTPUT_DIR / "adapter_model.safetensors"

    adapter_weights = load_file(
        str(adapter_path),
        device="cpu"
    )

    set_peft_model_state_dict(
        model,
        adapter_weights,
        adapter_name="default"
    )

    del adapter_weights
    del optimizer
    del scaler

    model.gradient_checkpointing_disable()
    model.config.use_cache = True

    torch.cuda.empty_cache()

    print("Оценка адаптированной модели...")

    adapted = evaluate(
        model,
        tokenizer,
        test
    )

    examples = []

    for before, after in zip(baseline, adapted):
        examples.append({
            "товар": before["товар"],
            "эталон": before["эталон"],
            "до_обучения": before["предсказание"],
            "после_обучения": after["предсказание"]
        })

    report = {
        "модель": MODEL_NAME,
        "метод": "LoRA",
        "обучающих_примеров": len(train),
        "проверочных_примеров": len(validation),
        "тестовых_примеров": len(test),
        "история": history,
        "до_обучения": score(baseline),
        "после_обучения": score(adapted),
        "примеры": examples
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"Результаты сохранены: {REPORT_PATH}")
    print(f"До обучения: {score(baseline)}")
    print(f"После обучения: {score(adapted)}")

    for row in examples[:5]:
        print(f"\nТовар: {row['товар']}")
        print(f"До обучения: {row['до_обучения']}")
        print(f"После обучения: {row['после_обучения']}")
        print(f"Эталон: {row['эталон']}")


if __name__ == "__main__":
    main()
