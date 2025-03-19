import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import SpikingVideoDataset
from clip.simple_tokenizer import SimpleTokenizer
from clip.model import CLIP
import gc

# 配置设备和路径
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "./weight/ucf101.pth"
TRAIN_JSON_PATH = "./train_set.json"
VAL_JSON_PATH = "./val_set.json"
SPIKE_H, SPIKE_W = 240, 320
BATCH_SIZE = 8
TOTAL_EPOCHS = 25
LEARNING_RATE = 1e-5
torch.cuda.empty_cache()

def initialize_model():
    model = CLIP(
        embed_dim=256,
        image_resolution=(SPIKE_H, SPIKE_W),
        vision_layers=(2, 2, 4, 2),
        vision_width=256,
        context_length=77,
        vocab_size=49408,
        transformer_width=128,
        transformer_heads=4,
        transformer_layers=4,
        input_channels=64
    ).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded model weights from {MODEL_PATH}", flush=True)
    else:
        print("Starting from scratch.", flush=True)
    return model

def create_data_loaders(batch_size=BATCH_SIZE):
    train_dataset = SpikingVideoDataset(
        json_file=TRAIN_JSON_PATH,
        spike_h=SPIKE_H,
        spike_w=SPIKE_W,
        device=DEVICE,
        target_frames=25
    )
    val_dataset = SpikingVideoDataset(
        json_file=VAL_JSON_PATH,
        spike_h=SPIKE_H,
        spike_w=SPIKE_W,
        device=DEVICE,
        target_frames=25
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader

def encode_texts(tokenizer, captions):
    tokenized_texts = []
    for caption in captions:
        tokenized = tokenizer.encode(caption)[:77]
        tokenized += [0] * (77 - len(tokenized))
        tokenized_texts.append(tokenized)
    return torch.tensor(tokenized_texts).to(DEVICE)

def plot_metrics(train_acc_i, train_acc_t, val_acc_i, val_acc_t, train_loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    epochs = range(1, TOTAL_EPOCHS + 1)

    ax1.plot(epochs, train_acc_i, label="Train I->T")
    ax1.plot(epochs, train_acc_t, label="Train T->I")
    ax1.plot(epochs, val_acc_i, label="Val I->T")
    ax1.plot(epochs, val_acc_t, label="Val T->I")
    ax1.set_title("Accuracy Metrics")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()

    ax2.plot(epochs, train_loss, label="Train Loss")
    ax2.plot(epochs, val_loss, label="Val Loss")
    ax2.set_title("Loss Metrics")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("./weight/training_metrics.png")
    plt.close()
    print("Metrics plot saved!", flush=True)

def train_and_validate(model, train_loader, val_loader):
    tokenizer = SimpleTokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4, betas=(0.9, 0.999))

    train_acc_i, train_acc_t = [], []
    val_acc_i, val_acc_t = [], []
    train_losses, val_losses = [], []

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        correct_i = correct_t = total = 0

        for idx, (spikes, texts) in enumerate(train_loader):
            spikes = spikes.to(DEVICE)
            text_tokens = encode_texts(tokenizer, texts)

            logits_per_image, logits_per_text = model(spikes, text_tokens)
            targets = torch.arange(len(spikes)).to(DEVICE)

            loss = (torch.nn.functional.cross_entropy(logits_per_image, targets) +
                    torch.nn.functional.cross_entropy(logits_per_text, targets)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            correct_i += (logits_per_image.argmax(1) == targets).sum().item()
            correct_t += (logits_per_text.argmax(1) == targets).sum().item()
            total += len(targets)

            # 实时输出训练进度
            if idx % 10 == 0:  # 每10个batch打印一次
                print(f"\rEpoch [{epoch+1}/{TOTAL_EPOCHS}] Batch [{idx+1}/{len(train_loader)}] "
                      f"Train Loss: {loss.item():.4f} | Acc: I->T {100 * correct_i / total:.2f}% T->I {100 * correct_t / total:.2f}%", end='', flush=True)

            del spikes, text_tokens, logits_per_image, logits_per_text
            torch.cuda.empty_cache()

        train_losses.append(epoch_train_loss / len(train_loader))
        train_acc_i.append(100 * correct_i / total)
        train_acc_t.append(100 * correct_t / total)

        # 完成一个epoch后换行
        print(f"\nEpoch {epoch+1}/{TOTAL_EPOCHS} completed.", flush=True)

        model.eval()
        epoch_val_loss = 0.0
        val_correct_i = val_correct_t = val_total = 0

        with torch.no_grad():
            for idx, (spikes, texts) in enumerate(val_loader):
                spikes = spikes.to(DEVICE)
                text_tokens = encode_texts(tokenizer, texts)

                logits_per_image, logits_per_text = model(spikes, text_tokens)
                targets = torch.arange(len(spikes)).to(DEVICE)

                loss = (torch.nn.functional.cross_entropy(logits_per_image, targets) +
                        torch.nn.functional.cross_entropy(logits_per_text, targets)) / 2

                epoch_val_loss += loss.item()
                val_correct_i += (logits_per_image.argmax(1) == targets).sum().item()
                val_correct_t += (logits_per_text.argmax(1) == targets).sum().item()
                val_total += len(targets)

                # 实时输出验证进度
                if idx % 10 == 0:  # 每10个batch打印一次
                    print(f"\rEpoch [{epoch+1}/{TOTAL_EPOCHS}] Batch [{idx+1}/{len(val_loader)}] "
                          f"Val Loss: {loss.item():.4f} | Acc: I->T {100 * val_correct_i / val_total:.2f}% T->I {100 * val_correct_t / val_total:.2f}%", end='', flush=True)

                del spikes, text_tokens
                torch.cuda.empty_cache()

        val_losses.append(epoch_val_loss / len(val_loader))
        val_acc_i.append(100 * val_correct_i / val_total)
        val_acc_t.append(100 * val_correct_t / val_total)

        # 打印最终的训练和验证结果
        print(f"\nEpoch {epoch+1}/{TOTAL_EPOCHS}")
        print(f"  Train Loss: {train_losses[-1]:.4f} | Acc: I->T {train_acc_i[-1]:.2f}% T->I {train_acc_t[-1]:.2f}%")
        print(f"  Val Loss: {val_losses[-1]:.4f}   | Acc: I->T {val_acc_i[-1]:.2f}% T->I {val_acc_t[-1]:.2f}%", flush=True)

        torch.save(model.state_dict(), MODEL_PATH)
        gc.collect()

    plot_metrics(train_acc_i, train_acc_t, val_acc_i, val_acc_t, train_losses, val_losses)

if __name__ == "__main__":
    model = initialize_model()
    train_loader, val_loader = create_data_loaders()
    train_and_validate(model, train_loader, val_loader)
