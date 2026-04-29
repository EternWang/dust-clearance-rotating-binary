.PHONY: all simulate figures animation test

all: simulate figures

simulate:
	python src/simulate_dust.py --n_particles 2000 --steps 2000 --dt 0.02 --seed 42 --save_every 20

figures:
	python src/make_figures.py

animation:
	python src/make_animation.py --max_particles 800 --fps 12 --dpi 110

test:
	python -m unittest discover -s tests
