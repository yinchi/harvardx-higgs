bookdown::render_book(output_format = "all")
file.rename('higgs.Rmd', 'docs/higgs.Rmd')
knitr::purl('docs/higgs.Rmd', 'docs/code.R')
