from pixel2flux import *

epic = 210707130
t,f,k,x,y = read_pixel(epic,4,'l')
labels = find_aper(t,f)
seg = draw_aper(f,labels,epic)
t,f = remove_nan(t,f)
bg,flags = get_bg(t,f,labels,epic)
ftot,xc,yc = plot_lc(t,f,labels,epic)