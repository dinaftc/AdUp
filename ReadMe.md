Our work represents a solution for test assignments in educational systems that defines and combines three learners' dimensions:
- Expected Performance: It is the expected performance of a learner for an assigned test.
- Aptitude: It is the learner's progression ability when assigned a test that is correctly completed.
- Learning Gap: It represents the past failures of the learner.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have python3  installed  
* You have installed requirements  `pip install -r requirements.txt`
* You have jupyter notebook installed

## Reproducibility of Experiments

Our experiments are divided into three research questions:
- RQ1: Is the combination of all dimensions well-adapted for attaining mastery and improving skill gain?
- RQ2.a: Do different settings of the skill update strategy exhibit different results?
- RQ2.b: Does the choice of the model of learner simulation impact mastery and skill gain?
- RQ3: Does an application of a meta-strategy that chooses to optimize a subset of dimensions at each iteration, improve mastery achievement?

To reproduce the reported results for these questions:
- Make sure to unzip the data file ```answers.csv``` located in ```./Data/matmat/```.
- Run the file `solution.py` to generate the results of RQ1. The results are generated for N=1, k=[3, 5].
  Results are saved in a CSV file. Find a copy file in ```./Results```: ```bkt_results_3_ncc_1.csv``` with k=3.
- Run the same file (`solution.py`) to generate the results of RQ2.a. Make sure before running to change the variable's value ```ncc_repeat``` to change the value of N to value 3.
  Results are also saved in a CSV file. Find a copy file in ```./Results```: ```bkt_results_3_ncc_3.csv``` with k=3.
- Run the file `solution_irt.py` to generate the results of RQ2.b. with the other simulation model: IRT.
  Results are also saved in a CSV file. Find a copy file in ```./Results```: ```irt_results_3_ncc_1.csv``` with k=3.
- Run the file `solution_mab.py` to generate the results of RQ3. The results are generated for N=1, k=3.
  Results are saved in a CSV file. Find a copy file in ```./Results```: ```bkt_results_3_ncc_1_mab.csv``` with k=3.

- Use ```graphs.ipynb``` to generate the different graphs presented in the paper using the results files in ```./Results```

## Citation
This work was publised in the DataEd@SIGMOD 2023:
```
@inproceedings{bouarour2023adaptive,
  title={Adaptive Test Recommendation for Mastery Learning},
  author={Bouarour, Nassim and Benouaret, Idir and d'Ham, C{\'e}dric and Amer-Yahia, Sihem},
  booktitle={Proceedings of the 2nd International Workshop on Data Systems Education: Bridging education practice with education research},
  pages={18--23},
  year={2023}
}
```

Its extension was published in the Transactions on Large Scale Data and Knowledge (TLSDK) Journal:
```
@incollection{bouarour2024multi,
  title={Multi-objective Test Recommendation for Adaptive Learning},
  author={Bouarour, Nassim and Benouaret, Idir and Amer-Yahia, Sihem},
  booktitle={Transactions on Large-Scale Data-and Knowledge-Centered Systems LVI: Special Issue on Data Management-Principles, Technologies, and Applications},
  pages={1--36},
  year={2024},
  publisher={Springer}
}
```
