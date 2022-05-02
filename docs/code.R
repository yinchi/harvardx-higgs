## ----Load-libraries----------------------------------------------------------------------------------------------

# R 4.1 key features: new pipe operator, \(x) as shortcut for function(x)
# R 4.0 key features: stringsAsFactors = FALSE by default, raw character strings r"()"
if (packageVersion('base') < '4.1.0') {
  stop('This code requires R >= 4.1.0!')
}

if(!require("pacman")) install.packages("pacman")
library(pacman)
p_load(tidyverse)


## ----Plot-example, fig.height=7, fig.width=7---------------------------------------------------------------------
anscombe %>% 
  gather(variable.x, value, -y1, -y2, -y3, -y4) |>
  select(variable.x = value) |>
  bind_cols(gather(anscombe, variable, value, -x1, -x2, -x3, -x4)) |>
  select(x = variable.x, variable, y = value) |>
  ggplot(aes(x,y)) + facet_wrap(~variable) +
  geom_point() + stat_smooth(method="lm",fullrange=T,se=F) +
  theme(aspect.ratio=1.)

