## Tuesday, Week 4

## meeting schedule
tuesday night
8:30 pm


## git practices
dont care

once in a while we copy stuff from overleaf to github as a backup

commit messages:
https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716


tag-name/task-kebab

everything is kebab case except for the slash between the name and task.
tag must be between 0 and 8 characters

example:
```
hllimpl-aaron/shared-memory-read
```



## notion or github issues
- github issues


## someone finds the original papers
- shove them into some branch
- make sure the filenames dont have illegal characters


## responsibilities
- tin: cpp
- sid: cpp
- aaron: stream generator
- matthias: python stream generator


## process communication
store it all in user streams without storing it
~~not some bindings https://github.com/pybind/pybind11~~
sockets is too slow
stream -> buffer (array) -> algorithm
use a ring buffer

write to buffer directly.
have the generator in c, and then call the generator in python.

use a seed.

cmake: aaron
ci: matthias


## meeting minutes
- forgot to ask


