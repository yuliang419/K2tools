from pixel2flux import *
from clean_lc import *

epic = 210843708
t,f,k,x,y = read_pixel(epic,4,'l')
labels = find_aper(t,f)
seg = draw_aper(f,labels,epic)
t,f = remove_nan(t,f)
bg,flags = get_bg(t,f,labels,epic)
t,ftot,xc,yc = plot_lc(t,f,labels,epic)
t,ftot,xc,yc = remove_thrust(t,ftot,xc,yc)
q = spline(t,ftot,xc,yc)