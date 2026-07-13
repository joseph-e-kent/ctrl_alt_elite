from __future__ import annotations

import argparse
import math
import random
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEFAULT_SEQ_LEN = 160
DEFAULT_STRIDE = 80
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 6
DEFAULT_EMBED_DIM = 192
DEFAULT_HIDDEN_DIM = 384
DEFAULT_NUM_LAYERS = 2
DEFAULT_LR = 1e-3
DEFAULT_TOP_K = 40
DEFAULT_TEMPERATURE = 0.9

AUTHOR_DIRS = {
	"lovecraft": BASE_DIR / "H.P.Lovecraft",
	"scooby": BASE_DIR / "scooby_doo",
}

TRANSLATION_TABLE = str.maketrans(
	{
		"\u2018": "'",
		"\u2019": "'",
		"\u201c": '"',
		"\u201d": '"',
		"\u2013": "-",
		"\u2014": "--",
		"\u2026": "...",
		"\u00a0": " ",
	}
)


@dataclass
class ModelConfig:
	seq_len: int = DEFAULT_SEQ_LEN
	embed_dim: int = DEFAULT_EMBED_DIM
	hidden_dim: int = DEFAULT_HIDDEN_DIM
	num_layers: int = DEFAULT_NUM_LAYERS


class CharTokenizer:
	def __init__(self, vocab: Sequence[str]) -> None:
		unique_vocab = list(dict.fromkeys(vocab))
		if "<unk>" not in unique_vocab:
			unique_vocab = ["<unk>"] + unique_vocab
		self.itos = unique_vocab
		self.stoi = {char: index for index, char in enumerate(self.itos)}

	@property
	def unk_index(self) -> int:
		return self.stoi["<unk>"]

	def encode(self, text: str) -> list[int]:
		normalized = normalize_text(text)
		return [self.stoi.get(character, self.unk_index) for character in normalized]

	def decode(self, token_ids: Sequence[int]) -> str:
		return "".join(self.itos[token_id] if 0 <= token_id < len(self.itos) else "" for token_id in token_ids)

	def to_dict(self) -> dict[str, object]:
		return {"itos": self.itos}

	@classmethod
	def from_dict(cls, data: dict[str, object]) -> "CharTokenizer":
		return cls(list(data["itos"]))


def normalize_text(text: str) -> str:
	normalized = unicodedata.normalize("NFKC", text)
	normalized = normalized.translate(TRANSLATION_TABLE)
	normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
	cleaned = []
	for character in normalized:
		if character == "\t":
			cleaned.append(" ")
		elif character == "\n" or 32 <= ord(character) <= 126 or ord(character) >= 160:
			cleaned.append(character)
		elif character.isspace():
			cleaned.append(" ")
	text = "".join(cleaned)
	while "\n\n\n" in text:
		text = text.replace("\n\n\n", "\n\n")
	while "  " in text:
		text = text.replace("  ", " ")
	return text.strip() + "\n"


def collect_texts(author_dir: Path) -> list[str]:
	texts: list[str] = []
	for file_path in sorted(path for path in author_dir.rglob("*") if path.is_file()):
		try:
			raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
		except OSError:
			continue
		normalized = normalize_text(raw_text)
		if normalized.strip():
			texts.append(normalized)
	if not texts:
		raise ValueError(f"No readable text files found in {author_dir}")
	return texts


def build_tokenizer(texts: Iterable[str]) -> CharTokenizer:
	vocab = sorted(set("".join(texts)))
	return CharTokenizer(vocab)


class CharWindowDataset(Dataset):
	def __init__(self, encoded_text: Sequence[int], seq_len: int, stride: int) -> None:
		if len(encoded_text) <= seq_len:
			raise ValueError("Corpus is too short for the chosen sequence length")
		self.encoded_text = torch.tensor(list(encoded_text), dtype=torch.long)
		self.seq_len = seq_len
		self.starts = list(range(0, len(encoded_text) - seq_len, stride))
		if not self.starts:
			self.starts = [0]

	def __len__(self) -> int:
		return len(self.starts)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		start = self.starts[index]
		window = self.encoded_text[start : start + self.seq_len + 1]
		return window[:-1], window[1:]


class CharGRU(nn.Module):
	def __init__(self, vocab_size: int, config: ModelConfig) -> None:
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, config.embed_dim)
		self.gru = nn.GRU(
			input_size=config.embed_dim,
			hidden_size=config.hidden_dim,
			num_layers=config.num_layers,
			batch_first=True,
			dropout=0.2 if config.num_layers > 1 else 0.0,
		)
		self.dropout = nn.Dropout(0.2)
		self.head = nn.Linear(config.hidden_dim, vocab_size)

	def forward(self, tokens: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
		embeddings = self.embedding(tokens)
		outputs, hidden = self.gru(embeddings, hidden)
		outputs = self.dropout(outputs)
		logits = self.head(outputs)
		return logits, hidden


def load_author_corpus(author: str) -> str:
	if author not in AUTHOR_DIRS:
		raise ValueError(f"Unknown author '{author}'. Expected one of: {', '.join(sorted(AUTHOR_DIRS))}")
	texts = collect_texts(AUTHOR_DIRS[author])
	return "\n\n".join(texts)


def make_dataset(author: str, seq_len: int, stride: int) -> tuple[CharTokenizer, CharWindowDataset, str]:
	corpus = load_author_corpus(author)
	tokenizer = build_tokenizer([corpus])
	encoded = tokenizer.encode(corpus)
	dataset = CharWindowDataset(encoded, seq_len=seq_len, stride=stride)
	return tokenizer, dataset, corpus


def split_dataset(dataset: Dataset, validation_fraction: float = 0.05) -> tuple[Dataset, Dataset]:
	val_size = max(1, int(len(dataset) * validation_fraction))
	train_size = max(1, len(dataset) - val_size)
	if train_size + val_size > len(dataset):
		train_size = len(dataset) - val_size
	return random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(42))


def train_one_author(
	author: str,
	*,
	epochs: int = DEFAULT_EPOCHS,
	batch_size: int = DEFAULT_BATCH_SIZE,
	seq_len: int = DEFAULT_SEQ_LEN,
	stride: int = DEFAULT_STRIDE,
	lr: float = DEFAULT_LR,
	config: ModelConfig | None = None,
	device: torch.device | None = None,
) -> Path:
	config = config or ModelConfig(seq_len=seq_len)
	tokenizer, dataset, corpus = make_dataset(author, seq_len=config.seq_len, stride=stride)
	train_dataset, val_dataset = split_dataset(dataset)

	use_cuda = torch.cuda.is_available() if device is None else device.type == "cuda"
	device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if use_cuda:
		torch.backends.cudnn.benchmark = True

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=use_cuda)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=use_cuda)

	model = CharGRU(len(tokenizer.itos), config).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()
	scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

	best_val = math.inf
	author_model_dir = MODEL_DIR / author
	author_model_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = author_model_dir / "first_version.pt"

	for epoch in range(1, epochs + 1):
		model.train()
		running_loss = 0.0
		for batch_inputs, batch_targets in train_loader:
			batch_inputs = batch_inputs.to(device, non_blocking=True)
			batch_targets = batch_targets.to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
				logits, _ = model(batch_inputs)
				loss = criterion(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			running_loss += loss.item()

		val_loss = evaluate(model, val_loader, criterion, device)
		train_loss = running_loss / max(1, len(train_loader))
		print(f"[{author}] epoch {epoch}/{epochs} train={train_loss:.4f} val={val_loss:.4f}")
		if val_loss < best_val:
			best_val = val_loss
			save_checkpoint(
				checkpoint_path,
				model=model,
				tokenizer=tokenizer,
				config=config,
				author=author,
				corpus_preview=corpus[:2000],
			)

	return checkpoint_path


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
	if len(loader) == 0:
		return math.inf
	model.eval()
	total_loss = 0.0
	total_batches = 0
	with torch.no_grad():
		for batch_inputs, batch_targets in loader:
			batch_inputs = batch_inputs.to(device, non_blocking=True)
			batch_targets = batch_targets.to(device, non_blocking=True)
			logits, _ = model(batch_inputs)
			loss = criterion(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))
			total_loss += loss.item()
			total_batches += 1
	return total_loss / max(1, total_batches)


def save_checkpoint(
	path: Path,
	*,
	model: nn.Module,
	tokenizer: CharTokenizer,
	config: ModelConfig,
	author: str,
	corpus_preview: str,
) -> None:
	payload = {
		"author": author,
		"config": asdict(config),
		"tokenizer": tokenizer.to_dict(),
		"model_state": model.state_dict(),
		"corpus_preview": corpus_preview,
	}
	torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device | None = None) -> tuple[str, CharTokenizer, ModelConfig, CharGRU]:
	checkpoint = torch.load(path, map_location=device or "cpu")
	tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
	config = ModelConfig(**checkpoint["config"])
	model = CharGRU(len(tokenizer.itos), config)
	model.load_state_dict(checkpoint["model_state"])
	model.to(device or torch.device("cpu"))
	model.eval()
	return checkpoint["author"], tokenizer, config, model


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
	if top_k <= 0 or top_k >= logits.numel():
		return logits
	values, _ = torch.topk(logits, top_k)
	cutoff = values[-1]
	filtered = logits.clone()
	filtered[filtered < cutoff] = float("-inf")
	return filtered


def generate_text(
	model: CharGRU,
	tokenizer: CharTokenizer,
	seed_text: str,
	*,
	target_chars: int = 1000,
	temperature: float = DEFAULT_TEMPERATURE,
	top_k: int = DEFAULT_TOP_K,
	device: torch.device | None = None,
) -> str:
	device = device or next(model.parameters()).device
	model.eval()

	seed_text = normalize_text(seed_text)
	prompt_ids = tokenizer.encode(seed_text)
	if not prompt_ids:
		prompt_ids = [tokenizer.unk_index]

	generated = list(prompt_ids)
	hidden = None

	with torch.no_grad():
		prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
		_, hidden = model(prompt_tensor)

		for _ in range(target_chars):
			input_token = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
			logits, hidden = model(input_token, hidden)
			next_logits = logits[0, -1] / max(temperature, 1e-6)
			next_logits = top_k_filter(next_logits, top_k)
			probabilities = torch.softmax(next_logits, dim=-1)
			next_token = torch.multinomial(probabilities, 1).item()
			generated.append(next_token)

	return tokenizer.decode(generated)


def resolve_checkpoint(author: str, checkpoint_arg: str | None) -> Path:
	if checkpoint_arg:
		return Path(checkpoint_arg)
	return MODEL_DIR / author / "first_version.pt"


def train_command(args: argparse.Namespace) -> None:
	authors = sorted(AUTHOR_DIRS) if args.author == "all" else [args.author]
	for author in authors:
		checkpoint = train_one_author(
			author,
			epochs=args.epochs,
			batch_size=args.batch_size,
			seq_len=args.seq_len,
			stride=args.stride,
			lr=args.lr,
		)
		print(f"Saved checkpoint to {checkpoint}")


def generate_command(args: argparse.Namespace) -> None:
	author = args.author
	checkpoint_path = resolve_checkpoint(author, args.checkpoint)
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the model first.")

	device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
	_, tokenizer, _, model = load_checkpoint(checkpoint_path, device=device)
	output = generate_text(
		model,
		tokenizer,
		args.seed,
		target_chars=args.chars,
		temperature=args.temperature,
		top_k=args.top_k,
		device=device,
	)
	print(output)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train and sample a character-level author style model.")
	subparsers = parser.add_subparsers(dest="command", required=True)

	train_parser = subparsers.add_parser("train", help="Train one or both author models.")
	train_parser.add_argument("--author", choices=["lovecraft", "scooby", "all"], default="all")
	train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
	train_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
	train_parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
	train_parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
	train_parser.add_argument("--lr", type=float, default=DEFAULT_LR)
	train_parser.set_defaults(func=train_command)

	generate_parser = subparsers.add_parser("generate", help="Generate a continuation from a seed sentence.")
	generate_parser.add_argument("--author", choices=["lovecraft", "scooby"], required=True)
	generate_parser.add_argument("--seed", required=True, help="Seed sentence or prompt.")
	generate_parser.add_argument("--chars", type=int, default=1000, help="Number of generated characters.")
	generate_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
	generate_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
	generate_parser.add_argument("--checkpoint", type=str, default=None)
	generate_parser.add_argument("--device", type=str, default=None)
	generate_parser.set_defaults(func=generate_command)

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	if args.command == "train":
		random.seed(42)
		torch.manual_seed(42)
	args.func(args)


if __name__ == "__main__":
	main()
