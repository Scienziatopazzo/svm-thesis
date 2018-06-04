# SVMRegression - Research Activity Log

## 4th June 2018
* Worked on figures.

## 3rd June 2018
* Implemented and ran `experiments4.py`.
* Started producing figures for the thesis.

## 31th May 2018
* Implemented and ran `experiments2.py` and `experiments3.py`.

## 30th May 2018
* Implemented the first set of experiments to be run multiple times in `experiments1.py`.
* Ran experiments 1. Todo: make experiments 2 about feature engineering combinations.

## 29th May 2018
* Written *Conclusions* and rewritten the *Introduction* to the thesis.

## 28th May 2018
* Finished *A Custom SVR Implementation* chapter draft.
* Completed *Results and Discussion* chapter draft.

## 27th May 2018
* Corrected error in Model 2 lagrangian, re-ran models. **It seems to be very good** (by c-index).
* Finished *Alternative SVR Models for Censored Datasets* chapter draft.
* Begun *A Custom SVR Implementation* chapter draft.
* Corrected various parts of thesis, included definitions of scoring metrics.

## 26th May 2018
* Almost finished *Alternative SVR Models for Censored Datasets* chapter draft.

## 17th May 2018
* Added subsections for *SVRC* and *RankSVMC*.

## 15th-16th May 2018
* Implemented corrections in thesis.
* Started *Alternative SVR Models for Censored Datasets* chapter in thesis.

## 9th May 2018
* Modified repeated holdout procedure in `training.py` so that the repetition happens before using the test set (added 'runs' attribute).
* Updated all relevant notebooks by running this version of holdout. Scores for R2 scoring are way worse, c-index scores are not. TODO: long computations with averaged test scores.

## 30th April 2018
* Implemented Model 2 from *Van Belle et al.* in `svrcens.py`. Had to fix a typo in the lagrangian dual specified in the paper. First test on dogs dataset yielded scores similar to those of other methods.

## 26th April 2018
* Implemented SVRC in `svrcens.py`. First test on dogs dataset yielded scores similar to those of other methods.
* Implemented RankSVMC (non simplified) in `svrcens.py`. The number of constraints make it too slow to run.
* Implemented RankSVMC with simplified constrainst in `svrcens.py`. First test on dogs dataset yielded scores slightly worse than those of other methods.
* Implemented Model 1 from *Van Belle et al.* in `svrcens.py`. First test on dogs dataset yielded scores similar to those of other methods.

## 25th April 2018
* Refactored svr code by splitting it into `svr.py` and `svrcens.py`.
* Introduced different scoring functions, implemented "c-index".
* Modified c-index from the original formulation, so that in the case of *y_pred[i] == y_pred[j]* the index doesn't automatically assume a correct ordering (because in this way a simple constant model would always obtain a perfect score).
* Modified `Custom-SVR-Censoring` tests so that c-index is used as the scoring metric. The original SVR and SVCR perform almost equally (with an apparently somewhat good score).

## 21st April 2018
* Tested customSVR (Standard) on dogs dataset. Results equal to sklearn svr.
* Added SVCR in `svr.py`, and restructured the file using an abstract class.
* Modified `training.py` to deal with censor deltas (so they are not scaled) (old method of scaling in `load_skl_dogs_2016` is untouched, needs to be adjusted).
* Created notebook `Custom-SVR-Censoring`.
* TODO: different scoring functions for SVRs, chooseable in gridsearch and testing.
* TODO: fix cases in which b cannot be computed.

## 17th April 2018
Completed draft of *Model Training and Selection*.

## 14th April 2018
* Completed draft of *Support Vector Machines for Regression* chapter in thesis.
* Updated *Introduction*.
* Started *Model Training and Selection* chapter in thesis.

## 12th April 2018
Written first two sections in *Support Vector Machines for Regression* chapter in thesis.

## 10th April 2018
Started *Support Vector Machines for Regression* chapter in thesis.

## 9th April 2018
Added *Introduction* stub and *Age* subsection in thesis.

## 8th April 2018
* Added an automatic Age fix to `load_df_dogs_2016`, since the values in the dataset were unusable.
* Used `ParameterGrid` in `training.py` to have more flexibility in parameter specification.
* Completed draft of *The Veterinary Dataset* chapter of thesis.

## 5th April 2018
* Added multiprocessing to `training.py`.
* Added outlier detection only on the training split in `training.py`.

## 4th April 2018
Started implementing scaling and doing outlier detection only on the training split of the data. TODO: outlier detection, proper pipeline.

## 3rd April 2018
* Experimented with polynomial kernel of deg>3, some cases make the computation too long, hanging the grid search.
* Experimented with scalers: `RobustScalers` retains outliers, hanging the grid search. `QuantileTranformer` obtains the same scores as the usual `StandardScaler`.
* Added the possibility of doing outlier detection and filtering on the dataset in `dogs_2006_2016.py`. Using this option seems to improve scores.

## 2nd April 2018
* Removed deprecated `set_value` calls in `dogs_2006_2016.py` and cleaned up the code.
* Cleaned up Training examples in notebooks, added a comparison between `GridSearchCV` and Repeated Houldout in the `Repeated Holdout` notebook.
* Modified thesis structure.

## 29th March 2018
Defined a first draft of the thesis structure.

## 25th March 2018
Implemented kernels in `svr.py` and nonlinear tests. Custom SVR consistently obtains the same scores of the Sklearn SVR in both linear and nonlinear tests.

**Note:** with the polynomial kernel, using negative values for `coef0` returns a Gurobi error. These values were removed from the holdout search, since they were seemingly never chosen anyways. Even though not specified in the sklearn documentation, a brief research seems to point at negative `coef0` values being unallowed in general. Strangely, Sklearn doesn't misbehave when fed those values.

## 23rd March 2018
Fixed a typo in `svr.py`, the SVR now performs very well.

## 22nd March 2018
Implemented a first version of a custom SVM Regression with Gurobi in `svr.py`. Did first tests on new notebook `Custom SVR Tests`, the SVM seems to perform poorly. Need to determine causes.

## 20th March 2018
Completed corrections of thesis draft.

## 20th November 2017
Continued work on thesis draft, reached "Censoring policies" in the "Preprocessing and exploration" section.

## 18th November 2017
Started working on first draft of thesis material in latex.

## 12th November 2017
Set up *Gurobi* and `gurobipy` and started experimenting with the solver, with the eventual goal of developing a modifiable SVM module and implementing the changes useful for dealing with censored data as described in *Van Belle et al*.

## 9th Nobember 2017
* Read *Van Belle et al, Support vector methods for survival analysis: a comparison between ranking and regression approaches*.
* Did 10 trials of holdout using both censoring policies in the `Repeated Holdout` notebook, then tested SVRs with the obtained hyperparameters. All with negative results.

## 6th November 2017
* Created `training` script with an implementation of *Holdout grid search*, together with the `Repeated Holdout` notebook. First experiments yielded negative results.

## 5th November 2017
* Updated `dogs_2006_2016.py`, removing uses of deprecated pandas method *set_value*.

## 29th October 2017
* Added `Preprocessing` check for *Age*. When computing age as *(death-birth)* almost all values were erroneous. When doing it as *(firstvisit-birth)* (age at the first visit) most values were correct but there were still 50 errors. The origin of the errors is unclear, as many erroneous values appear to have no relation with the provided dates.
* Created notebook `Feature engineering`:
  - Added new feature *Therapy to visit*, a time delta in days from the beginning of the therapy to the first visit. (By adding this measure to *Survival time* it is possible to obtain a survival time from the beginning of the therapy, thus fulfilling the objective of 21/10/17).
  - Tried out removal of highly correlated features in various combinations.
  - Explored the relationship between *IP Gravity* and *Vrig Tric* when both values are different from 0. IP Gravity is just a discretization of Vrig Tric, so there's no point in using both.
* Reviewed k-fold cross-validation. Started using more folds as using 2 was clearly bad practice. Benefits of using holdout over it (as instructed) are unclear.

## 28th October 2017
* Introduced filling of NA values in notebook `NA fill`, using different policies:
  - *mean* -  Using the mean of the feature.
  - *normal* - Generating values from a normal distribution modeled after the feature (checked approximate normality with a qqplot beforehand).
  - **TODO:** Using values predicted from a model (eg. linear regression) trained with correlated features.
* Introduced primitive methods for dealing with the right-censoring in *Survival time* in notebook `Censoring`:
  - *drop* - just drop all data for which *Dead*==0.
  - *max* - when *Dead*==0, replace all *Survival time* values with the maximum Survival time of data for which *Dead*==1.

Combining the *normal* NA policy with *drop* censoring policy an SVR with grid search was able to obtain a slightly positive score, although not stable.

## 25th October 2017
* New information about the dataset:
  - Obtained description of *isachc* scale. *C* means that the dog was only classified with the newer scale, so it's equivalent to *NA*. As suggested, the feature was deemed probably irrelevant.
  - Resolved issue posed by the fact that all dogs had a death date, even alive ones. The death date is in fact right-censored for dogs that were last recorded as alive.

## 23rd October 2017
Added attribute *fixErrors* to *load_df_dogs_2016* in `dogs_2006_2016.py`, for fixing the one wrong *Survival time* value in the dataset.

## 22nd October 2017
* Conducted first exploratory analysis of the dogs dataset in `Dataset Analysis` notebook.
  - The only visible correlations with *Survival time* detected were with *Age* and *Asx/ao* (and in minor proportion *FE %*)
* Tried out various SV regressions in `Dataset Analysis` notebook, all with awful scores.
* *"Solved"* the apparent issue exposed on 21/10/17: some date entries were automatically transcribed by excel into the American format, but they are clearly recognizable by `converters.py` as such, so data consistency is maintained.

## 21st October 2017
* Finished first draft of `Preprocessing` notebook. Notes:
  - All detected date inconsistencies are related to therapies started before the first visit. Could possibly refer to therapies started with other clinics. Therefore potentially not an error, although the current survival time measure does not take this into account and it may be useful to try computing one that does.
  - A single error of 365 days in survival time was detected. Survival time values were written by hand, so the detected error was probably the result of misreading the year of an entry.
  ~~**Note:** this means that if the entirety of the "Date of death" column uses American date formats, then whoever transcribed "Survival time" read the wrong value for every date that can be also read as European. (Our program only interprets dates with the American format if they can't be understood as European)~~ (not an issue, look at 22/10/17)
  - No inconsistencies found for *cardiac arrest death -> death* and the *Therapy category* value
* Tried out Sklearn SVR module with matplotlib in `Sklearn` notebook.
* Created `Dataset Analysis` notebook.

## 20th October 2017
Started working on preprocessing the dogs dataset.

## 10th-17th October 2017
Read *Smola and Schölkopf, A Tutorial on Support Vector Regression*.

## August-September 2017
Read *Burges, A Tutorial on Support Vector Machines for Pattern
Recognition*.
