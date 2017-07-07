PYTHON_BIN_DIR = /usr/local/python35/bin

errors:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON_BIN_DIR)/pylint -j 4 -E *.py
.PHONY: errors

mypy:
	PYTHONPATH=$(PYTHONPATH) MYPYPATH="/home/$${USER}/ATB_ONE" $(PYTHON_BIN_DIR)/mypy --fast-parser $$(find . -name '*.py' |  sed "s|^\./||")
.PHONY: mypy
