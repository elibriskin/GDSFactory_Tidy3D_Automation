import numpy as np
import gdsfactory as gf
from gdsfactory.generic_tech import LAYER_STACK, get_generic_pdk
import gplugins as gp
import gplugins.tidy3d as gt
from gplugins import plot
from gplugins.common.config import PATH
import matplotlib.pyplot as plt
import gdstk


import tidy3d as td
import tidy3d.web as web
from tidy3d.plugins.mode import ModeSolver

def get_port_coords(component, port_name, buffers=(0, 0, 0)):
    '''
    Obtain the coordinates of a given port of a photonic component.
    '''
    return component.ports[port_name].center[0] + buffers[0], component.ports[port_name].center[1]+ buffers[1], 0+ buffers[2]


def create_port_monitors(component, ports, monitor_type, size, freqs, mode_spec=None):
    '''
    Creates a series of monitors situated at the ports of a give photonic component.
    '''
    #List of port monitors
    port_monitors = []

    #Iterate through each specified port
    for port in ports:

        if monitor_type == "flux":

            #Create flux monitor centered at port coordinates
            port_monitor = td.FluxMonitor(
                center=get_port_coords(component, port.name),
                size=size,
                freqs=freqs,
                name=f"{monitor_type}_port_{port.name}",
            )
        elif monitor_type == "mode":

            #Create mode monitor centered at port coordinates
            port_monitor = td.ModeMonitor(
                center=get_port_coords(component, port.name),
                size=size,
                freqs=freqs,
                mode_spec=mode_spec,
                name=f"{monitor_type}_port_{port.name}",
            )
        else:
            raise Exception("Invalid monitor type! Valid monitor types are 'flux' and 'mode")
        
        #Add to list of port monitors
        port_monitors.append(port_monitor)

    return port_monitors


def convert_to_structure(component, 
                         reference_plane="bottom",
                         axis = 2,
                         slab_bounds=(0, .22),
                         medium_wg = td.Medium(permittivity=3.48**2),
                         medium_sub = td.Medium(permittivity=1.45**2),
                         add_substrate=False,
                         substrate_height=.5):
    '''
    Takes in a GDSFactory defined p-cell and converts it into a structure that can be simulated
    in Tidy3D.
    '''

    #Flatten all cells into single p-cell
    component.flatten()

    #Create filename for saving GDS
    filename = f'{component.name}.gds'

    #Save file
    component.write_gds(filename)

    #Load in GDS component
    lib_loaded = gdstk.read_gds(filename)

    #Create component geometry
    component_geo = td.Geometry.from_gds(
        lib_loaded.cells[1],
        gds_layer=lib_loaded.cells[1].polygons[1].layer,
        gds_dtype=lib_loaded.cells[1].polygons[1].datatype,
        axis=axis,
        slab_bounds=slab_bounds,
        reference_plane=reference_plane

    )

    #Convert component geometry to structure
    component_structure = td.Structure(geometry=component_geo, medium=medium_wg)

    #Return discrete p-cell structure without substrate
    if add_substrate == False:
        return [component_structure]
    else:
        
        #Create a substrate the size of bounding box of p-cell
        substrate = td.Structure(
            geometry = td.Box(
                center=(component_geo.bounding_box.center[0], component_geo.bounding_box.center[1], -substrate_height/2),
                size=(component_geo.bounding_box.size)
            ),
            medium=medium_sub,
            name="Substrate"
        )

        #Return both component structure and substrate structure
        structures = [substrate, component_structure]
        return structures

