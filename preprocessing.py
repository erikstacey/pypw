import numpy as np
import config
import matplotlib.pyplot as pl


def flux_to_mag_e(data, err):
    mag_data = -2.5 * np.log10(data)
    mag_err = 2.5 * err / (np.log(10) * data)
    return mag_data, mag_err


def stripgaps(time_in, data_in, err_in, gapsize, points):
    """Removes points from around gaps in time of gapsize from a time, data, err dataset"""
    # remove points around start and end
    time, data, err = time_in[points:-points], data_in[points:-points], err_in[points:-points]
    array_mask = np.ones(len(time), dtype=bool)
    for i in range(len(time) - 1):
        if abs(time[i] - time[i + 1]) >= gapsize:
            # handle near start case
            if i < points:
                array_mask[:i] = False
                array_mask[i:i + points] = False
            # and end case
            elif len(time) - i < points:
                array_mask[i - points:i] = False
                array_mask[i:] = False
            else:
                array_mask[i - points:i + points] = False
    time_final, data_final, err_final = time[array_mask], data[array_mask], err[array_mask]

    return time_final, data_final, err_final


def preprocess(time0, data0, err0):
    time, data, err = time0, data0, err0
    if config.strip_gaps:
        time, data, err = stripgaps(time, data, err, config.gap_size, config.strip_gaps_points)

    if config.output_in_mmag and config.target_dtype != "mmag":
        if config.target_dtype == "flux" and config.dtype == "flux":
            datamean = np.mean(data)
            return time, data-datamean, err, datamean
        else:
            print("Can only convert outputs to mmag if data is in flux - check target_dtype and dtype")

    if config.target_dtype == "flux" and config.target_dtype != config.dtype:
        print("Program incapable of converting something else to flux at the moment")
        quit()
    elif config.target_dtype == config.dtype:
        return time, data-np.mean(data), err, None
    elif config.target_dtype == "mag" and config.dtype == "flux":
        if not config.quiet:
            print("Successfully converted data from flux to mag")
        data, err = flux_to_mag_e(data, err)
    elif config.target_dtype == "mmag" and config.dtype == "flux":
        if not config.quiet:
            print("Successfully converted data from flux to mmag")
        data, err = flux_to_mag_e(data, err)
        data, err = data * 1000, err * 1000
    elif config.target_dtype == "mmag" and config.dtype == "mag":
        if not config.quiet:
            print("Successfully converted data from mag to mmag")
        data, err = data * 1000, err * 1000
    else:
        print("Could not figure out how to convert to target data type")
        quit()
    return time, data-np.mean(data), err, None


if __name__ == "__main__":
    time1 = np.linspace(0, 4, 40)
    time2 = np.linspace(12, 25, 130)
    time3 = np.linspace(28, 35, 70)
    testtime = np.append(time1, time2)
    testtime = np.append(testtime, time3)

    data = np.sin(2 * np.pi * (0.2 * testtime))
    err = np.ones(len(testtime))

    pl.plot(testtime, data, marker='.', linestyle='none', markersize=3, color='red')
    testtime_p, data_p, err_p = stripgaps(testtime, data, err, 0.5, 30)
    pl.plot(testtime_p, data_p, marker='.', linestyle='none', markersize=3, color='black')
    pl.show()
