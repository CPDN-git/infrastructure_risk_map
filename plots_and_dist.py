import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns




#################################################

# Class for remapping seaborn plots into subplots, seaborn hardcoded to make new fig each time

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

#################################################################


array_link='/home/users/jburns59/cpdn_analysis/examples/'

tmax_hist=np.load(array_link+'batch_646_data_item3236_daily_maximum.npy')
wind_hist=np.load(array_link + 'batch_646_data_item3249_daily_mean.npy')


tmax_15=np.load(array_link+'batch_647_data_item3236_daily_maximum.npy')
wind_15=np.load(array_link + 'batch_647_data_item3249_daily_mean.npy')


tmax_2=np.load(array_link+'batch_648_data_item3236_daily_maximum.npy')
wind_2=np.load(array_link + 'batch_648_data_item3249_daily_mean.npy')



tmax_hist_shape_init=tmax_hist.shape
wind_hist_shape_init=wind_hist.shape

for i in range(90):
	wind_hist=np.delete(wind_hist,i)

wind_hist=np.reshape(wind_hist,tmax_hist_shape_init)



tmax_2_shape_init=tmax_2.shape

for i in range(90):
	wind_2=np.delete(wind_2,i)

wind_2=np.reshape(wind_2,tmax_2_shape_init)


print tmax_2.shape
print wind_2.shape


wind_min=[]
wind_min=[np.amin(wind_hist),np.amin(wind_15),np.amin(wind_2)]

wind_min=np.amin(wind_min)

wind_max=[]
wind_max=[np.amax(wind_hist),np.amax(wind_15),np.amax(wind_2)]
wind_max=np.amax(wind_max)

print('minimum wind is ' + str(wind_min) + ' m/s') 
print wind_max

tmax_min=[]
tmax_min=[np.amin(tmax_hist),np.amin(tmax_15),np.amin(tmax_2)]
tmax_min=np.amin(tmax_min)

tmax_max=[]
tmax_max=[np.amax(tmax_hist),np.amax(tmax_15),np.amax(tmax_2)]
tmax_max=np.amax(tmax_max)

print tmax_min
print tmax_max


plot_hist=sns.jointplot(x=np.array(wind_hist), y=np.array(tmax_hist),xlim=(0,15),ylim=(280,320),kind="hex", color="k");
plot_hist.set_axis_labels('Average Daily Wind Speed m/s','Max Ambient Temp  K')

plot_15=sns.jointplot(x=np.array(wind_15), y=np.array(tmax_15),xlim=(0,15),ylim=(280,320), kind="hex", color="k");
plot_15.set_axis_labels('Average Daily Wind Speed m/s','Max Ambient Temp  K')


plot_2=sns.jointplot(x=np.array(wind_2), y=np.array(tmax_2),xlim=(0,15),ylim=(280,320), kind="hex", color="k");
plot_2.set_axis_labels('Average Daily Wind Speed m/s','Max Ambient Temp  K')

fig = plt.figure()
gs=gridspec.GridSpec(1,3)

mg0 = SeabornFig2Grid(plot_hist,fig,gs[0])
mg1 = SeabornFig2Grid(plot_15,fig,gs[1])
mg2 = SeabornFig2Grid(plot_2,fig,gs[2])

gs.tight_layout(fig)

plt.show()

