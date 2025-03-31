# Get THEMIS plasma parameters
import pyspedas
from pytplot import get_data
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

time_range = ['2025-03-02','2025-03-16']
sc = 'b' # Denotes which THEMIS probe to load data from
# Electrostatic Analyzer Reduced mode - 3sec resolution, low angular resolution
esa_redu = pyspedas.themis.esa(probe=sc, level='l2', trange=time_range,
                                varnames=['th'+sc+'_peir_en_eflux',
                                          'th'+sc+'_peir_density',
                                          'th'+sc+'_peir_avgtemp',
                                          'th'+sc+'_peir_velocity_gse'], no_update=True)
# Fluxgate Magnetometer Spin mode
fgm_spin = pyspedas.themis.fgm(probe=sc, level='l2', trange=time_range,
                                varnames=['th'+sc+'_fgs_btotal',
                                           'th'+sc+'_fgs_gsm'], no_update=True)

# ion flux spectrum
i_redu_flux = get_data(esa_redu[0])
i_redu_flux_cm = i_redu_flux.y
esa_time = i_redu_flux.times
# ion density 
i_density = get_data(esa_redu[1])
i_n_cm = i_density.y
# ion temperature
i_temp = get_data(esa_redu[2])
i_temp_eV = i_temp.y
# ion velocity
i_redu_vel = get_data(esa_redu[3])
i_redu_vel_gse = i_redu_vel.y
print(i_redu_vel_gse.shape)
# Extract each column into x,y,z components
i_redu_vel_x, i_redu_vel_y, i_redu_vel_z = i_redu_vel_gse.T

esa_redu_df = pd.DataFrame({'density': i_n_cm, 'temp': i_temp_eV, 'u_x': i_redu_vel_x,
                       'u_y': i_redu_vel_y, 'u_z': i_redu_vel_z})
print(esa_redu_df.head())

fig, ax = plt.subplots(3, 1, sharex=True)
esa_redu_df['density'].plot(ax=ax[0], color='blue', label='Density (cm^-3)')
esa_redu_df['temp'].plot(ax=ax[1], color='red', label='Temperature (eV)')
esa_redu_df['u_x'].plot(ax=ax[2], color='green', label='u_x')
esa_redu_df['u_y'].plot(ax=ax[2], color='orange', label='u_y')
esa_redu_df['u_z'].plot(ax=ax[2], color='purple', label='u_z')
ax[0].set_ylabel('Density (cm^-3)')
ax[1].set_yscale('log')
ax[1].set_ylabel('Temperature (eV)')
ax[2].set_ylabel('Velocity (km/s)')
ax[2].set_xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()

# Extra data products; different spacecraft measurement modes
# Full mode - ~3min resolution, high angular resolution
# esa_full = pyspedas.themis.esa(probe=sc, level='l2', trange=time_range,
#                                 varnames=['th'+sc+'_peif_en_eflux',
#                                           'th'+sc+'_peif_density',
#                                           'th'+sc+'_peif_avgtemp'], no_update=False)
# Burst mode - short bursts, high time res., high angular res.
# esa_burs = pyspedas.themis.esa(probe=sc, level='l2', trange=time_range,
#                                 varnames=['th'+sc+'_peib_en_eflux',
#                                           'th'+sc+'_peib_density',
#                                           'th'+sc+'_peib_avgtemp'], no_update=False)
# fgm_engi = pyspedas.themis.fgm(probe=sc, level='l2', trange=time_range,
#                                 varnames=['th'+sc+'_fge_btotal',
#                                            'th'+sc+'_fge_gsm'])
# fgm_high = pyspedas.themis.fgm(probe=sc, level='l2', trange=time_range,
#                                 varnames=['th'+sc+'_fgh_btotal',
#                                            'th'+sc+'_fgh_gsm'])
# fgm_lowr = pyspedas.themis.fgm(probe=sc, level='l2', trange=time_range,
#                                 varnames=['th'+sc+'_fgl_btotal',
#                                            'th'+sc+'_fgl_gsm'])