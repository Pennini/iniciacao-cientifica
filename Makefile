.PHONY: help prepare train-har train-patchtst compare

PYTHON ?= python

help:
	@echo "Comandos disponíveis:"
	@echo "  make prepare        - Prepara os dados (src.cli prepare)"
	@echo "  make train-har      - Treina e avalia o HAR (src.cli train-har)"
	@echo "  make train-patchtst - Treina e avalia o PatchTST (src.cli train-patchtst)"
	@echo "  make compare        - Compara HAR vs PatchTST (src.cli compare)"

prepare:
	$(PYTHON) -m src.cli prepare

train-har:
	$(PYTHON) -m src.cli train-har

train-patchtst:
	$(PYTHON) -m src.cli train-patchtst

compare:
	$(PYTHON) -m src.cli compare
