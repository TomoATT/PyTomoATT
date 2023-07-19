# module for visualization functions
import geoviews as gv
import geoviews.feature as gf
from geoviews import opts
import numpy as np
from holoviews.operation.datashader import rasterize, spread


def plot_srcrec(SR, weight=False, fname=None):
    """
    Plot source and receiver locations
    SR: source and receiver object
    weight: if plots weight of the source/receiver
    fname: if not None, save the plot to a file
    """

    gv.extension('bokeh')

    #tiles = gv.tile_sources.Wikipedia()
    #tiles = gv.tile_sources.StamenTerrain()
    #tiles = gv.tile_sources.EsriReference()
    #tiles = gv.tile_sources.EsriUSATopo()
    #tiles = gv.tile_sources.OSM()*gv.feature.coastline()
    tiles = gv.feature.coastline(scale='50m')

    if weight==False:
        # source points
        SR.src_points['_evdp'] = SR.src_points['evdp'] # for protting
        ds = gv.Dataset(SR.src_points, kdims=['evlo','evla','evdp','_evdp'], vdims=['num_rec'])

        lola = rasterize(gv.Points(ds, kdims=['evlo','evla'],  vdims=['num_rec'])).opts(cmap='viridis')
        lod  = rasterize(gv.Points(ds, kdims=['evlo','evdp'],  vdims=['num_rec'])).opts(cmap='viridis')
        lad  = rasterize(gv.Points(ds, kdims=['_evdp','evla'], vdims=['num_rec'])).opts(cmap='viridis')     #.opts(invert_yaxis=True)

        #lola = spread(rasterize(gv.Points(ds, kdims=['evlo','evla'],  vdims=['num_rec']), aggregator="sum").opts(cmap='viridis'))
        #lod  = spread(rasterize(gv.Points(ds, kdims=['evlo','evdp'],  vdims=['num_rec']), aggregator="sum").opts(cmap='viridis'))
        #lad  = spread(rasterize(gv.Points(ds, kdims=['_evdp','evla'], vdims=['num_rec']), aggregator="sum").opts(cmap='viridis'))     #.opts(invert_yaxis=True)

        # station points
        SR.count_events_per_station()
        # add '_stdp' column for plotting
        SR.rec_points['_stdp'] = -0.001*SR.rec_points['stel'] # converting elevation [m] to depth [km]
        df_sta = SR.rec_points.groupby(['staname']).apply(lambda x: x.iloc[-1])
        r_lola = gv.Points(df_sta, kdims=['stlo', 'stla'], vdims=['num_events','staname']).opts(color='red', size=2, tools=['hover'],)
        r_lod = gv.Points(df_sta, kdims=['stlo', '_stdp'], vdims=['num_events','staname']).opts(color='red', size=2, tools=['hover'],)
        r_lad = gv.Points(df_sta, kdims=['_stdp','stla'], vdims=['num_events','staname']).opts( color='red', size=2, tools=['hover'],)

        # fix layout
        layout=(lola.opts(width=500, height=500)*tiles*r_lola +
                 lad.opts(width=200,height=500)*r_lad +
                 lod.opts(width=500,height=200, invert_yaxis=True)*r_lod).cols(2).opts(title='Nevada+SC+NC')

        if fname is not None:
            gv.save(layout, fname+'.html')

        return layout

    else:
        # weight plot

        # source points
        SR.src_points['_evdp'] = SR.src_points['evdp'] # for protting
        ds = gv.Dataset(SR.src_points, kdims=['evlo','evla','evdp','_evdp'], vdims=['weight'])

        # station points
        SR.count_events_per_station()
        # add '_stdp' column for plotting
        SR.rec_points['_stdp'] = -0.001*SR.rec_points['stel'] # converting elevation [m] to depth [km]
        df_sta = SR.rec_points.groupby(['staname']).apply(lambda x: x.iloc[-1])

        # plot

        # show min and max weight
        print("min weight = ", np.min(SR.src_points['weight']))
        print("max weight = ", np.max(SR.src_points['weight']))

        w_evs = spread(rasterize(gv.Points(ds, kdims=['evlo','evla'], vdims=['weight']), aggregator="mean").opts( cmap='plasma'), how='source').opts(width=500, height=500, colorbar=True, tools=['hover'], colorbar_position='bottom')
        w_sta = spread(rasterize(gv.Points(df_sta, kdims=['stlo', 'stla'], vdims=['weight']),aggregator="mean").opts(cmap='viridis'), px=2, how='source').opts(width=500, height=500, colorbar=True, tools=['hover'], colorbar_position='bottom')

        # log scale for colorbar
        w_evs.opts(cmap='plasma', colorbar=True, logz=True)
        w_sta.opts(cmap='viridis', colorbar=True, logz=True)

        if fname is not None:
            gv.save((w_evs*tiles + w_sta*tiles).cols(2), fname+'.html')

        return (w_evs*tiles + w_sta*tiles).cols(2)
