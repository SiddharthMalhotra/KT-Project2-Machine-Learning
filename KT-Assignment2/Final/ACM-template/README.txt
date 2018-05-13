Skeleton files for a standard-format ACM paper. Start with these,
and follow the hints, and you'll be in good shape.

Note that the Makefile just makes things happen, doesn't look at any
timestamps etc.

make:
	will run pdflatex once

make bbl:
	will run bibtex as well as pdflatex

make final:
	will create a named output.tex and matching output.pdf file,
	with all latex source code in a single file, as required by
	ACM

Alistair Moffat,
August 2017.
