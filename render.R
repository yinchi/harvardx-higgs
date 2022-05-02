bookdown::render_book(output_format = "all")
file.rename('merged.Rmd', 'docs/merged.Rmd')
knitr::purl('docs/merged.Rmd', 'docs/code.R')
