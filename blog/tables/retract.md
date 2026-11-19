| claim we made | what was wrong | corrected |
|---|---|---|
| 12-trial Optuna studies had converged | seeded TPE spends its first 10 trials on random draws, so three traces returned identical parameters and a rank correlation of 1.000 | continued to 26 to 34 trials; every test MAE improved |
| three traces re-tuned better after a refit | the scorer reused window files older than the model, so the same scores were reported under new parameters, identical to 16 digits | rescoring is keyed on the model's mtime |
| `single_row` scored a perfect curve | it reads `eval_rtilde` for the block it scores, so its own bin decides the hit | it is an oracle, reported as one, never as a competitor |
