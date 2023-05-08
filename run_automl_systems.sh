
ROOT="$(cd -- "$MY_PATH" && pwd)"
echo "NOW IN $ROOT"

# tpot
source "/home/hk/python_venvs/37_tpot/bin/activate"
echo "Activated TPOT's venv"
cd "_tpot"
echo "NOW IN" $(pwd)
python run_tpot.py
echo "Ran run_tpot.py"
deactivate
cd "$ROOT"
echo "NOW IN" $(pwd)



# dswizard

source "/home/hk/python_venvs/dswizard38/bin/activate"
echo "Activated DSWIZARD's venv"
cd "_dswizard"
echo "NOW IN" $(pwd)
python run_dswizard.py
echo "Ran run_dswizard.py"
deactivate
cd "$ROOT"
echo "NOW IN" $(pwd)




# Alphad3m
source "/home/hk/python_venvs/37_alphad3m/bin/activate"
echo "Activated AlphaD3M's venv"
cd "_alphad3m"
echo "NOW IN" $(pwd)
python run_alphad3m.py
echo "Ran run_alphad3m.py"
deactivate
cd "$ROOT"
echo "NOW IN" $(pwd)


# Autosklearn
source "/home/hk/python_venvs/autosklearn37/bin/activate"
echo "Activated Autosklearn's venv"
cd "_auto-sklearn"
echo "NOW IN" $(pwd)
python run_autosklearn1.py
echo "Ran run_autosklearn1.py"
python run_autosklearn2.py
echo "Ran run_autosklearn2.py"
cd "$ROOT"
echo "NOW IN" $(pwd)
