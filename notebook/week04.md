

questions:
- depth of mathematicalness required
- deterministic programs hard to test, what should we do (apart from hypothesis testing and reordering + repeating streams)
- how to compare results, i.e., whether that is enough / what we actually submit
- ways to generate data (in consideration of large sizes)
- miscellaneous extensions
- random / structured data?




### depth of mathematicalness required
- give a sketch of the proof
- convincing enough for a 3821 student
- if important to project, include the entire proof (even if it is mostly copied from the paper with citation)
- temper the details depending on the importance of the algorithm
- even if not important, but relevant to flow of project (e.g., intermediate result), is fine to leave it in as a sketch / read this paper



### probabilistic programs hard to test, what should we do (apart from hypothesis testing and reordering + repeating streams)

- monte carlo estimation
- variance in response is resolved by increasing trials
- plot distribution of times for a fixed seeds + plot distribution of cardinality estimates

- implementation correctness is measured by closeness to theoretical results
- like provided not more than a few orders of magnitude off, then is fine

- hesitantly say that accidentally being correct
- proof is better than testing. in the absence of proof, many tests are also fine.


### how to compare results, i.e., whether that is enough / what we actually submit

- if it takes 10 actual weeks to do blackbox testing, it is probably fine (because we will likely run into some hiccups in implementation) more stuff would be better but extensiony
- also we should start early on synthesising stuff to write into the report.


### ways to generate data (in consideration of large sizes)

- synthetic data is fine, and we have some guarantees on the data we can exploit.
- try to find comp2041 fake student enrolment data.
- we want to show that our algorithm actually works on real data.
- https://www.kaggle.com/datasets?sizeStart=100%2CGB


### miscellaneous extensions

- give/implement a use case for CE
- case study


### whats in the writeup

- writeup is for showing results, like a paper
- do it like textbooks to onboard students onto something, and which give practical use cases for stuff
