/*
 * Copyright (C) 2008, 2009, 2010, 2012 Richard Membarth <richard.membarth@cs.fau.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <inttypes.h>
#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>

#include "gimp_main.hpp"
#include "gimp_gui.hpp"
#include "multi_res_cpu.hpp"
#include "multi_res_host.hpp"


static void query       (void);
static void run         (const gchar      *name,
                         gint              nparams,
                         const GimpParam  *param,
                         gint             *nreturn_vals,
                         GimpParam       **return_vals);

void run_multi_filter(GimpDrawable *drawable, GimpPreview  *preview);

/* Set up default values for options */
filter_config filter_vals = {
    3,  /* geometric spread */
    3,  /* photometric spread */
    1,  /* preview */
    1,  /* GPU */
    0   /* use float */
};

GimpPlugInInfo PLUG_IN_INFO = {
    NULL,
    NULL,
    query,
    run
};

MAIN()

static void query(void) {
    static GimpParamDef args[] = {
        {
            GIMP_PDB_INT32,
            (gchar *) "run-mode",
            (gchar *) "Run mode"
        },
        {
            GIMP_PDB_IMAGE,
            (gchar *) "image",
            (gchar *) "Input image"
        },
        {
            GIMP_PDB_DRAWABLE,
            (gchar *) "drawable",
            (gchar *) "Input drawable"
        },
        {
            GIMP_PDB_INT32,
            (gchar *) "sigma_d",
            (gchar *) "Geometric spread"
        },
        {
            GIMP_PDB_INT32,
            (gchar *) "sigma_r",
            (gchar *) "Photometric spread"
        },
        {
            GIMP_PDB_INT32,
            (gchar *) "gpu",
            (gchar *) "Run on GPU"
        },
        {
            GIMP_PDB_INT32,
            (gchar *) "use_float",
            (gchar *) "Use floats for higher accuracy"
        }
    };

    gimp_install_procedure(
        "plug-in-multi-res-filter",
        "Multiresolution gradient adaptive filter (nonlinear) with bilateral kernel filter...",
        "Removes noise from the image while preserving edges",
        "Richard Membarth (richard.membarth@cs.fau.de)",
        "Richard Membarth",
        "10/07/2008",
        "_Multiresolution gradient adaptive filter...",
        "GRAY*",
        GIMP_PLUGIN,
        G_N_ELEMENTS(args), 0,
        args, NULL);

    gimp_plugin_menu_register("plug-in-multi-res-filter", "<Image>/Filters/Enhance");
}


static void run(const gchar *name, gint nparams, const GimpParam *param, gint *nreturn_vals, GimpParam **return_vals) {
    static GimpParam  values[1];
    GimpPDBStatusType status = GIMP_PDB_SUCCESS;
    GimpRunMode       run_mode;
    GimpDrawable     *drawable;

    /* Setting mandatory output values */
    *nreturn_vals = 1;
    *return_vals  = values;

    values[0].type = GIMP_PDB_STATUS;
    values[0].data.d_status = status;

    /* Getting run_mode - we won't display a dialog if
     * we are in NONINTERACTIVE mode */
    run_mode = (GimpRunMode) param[0].data.d_int32;

    /*  Get the specified drawable  */
    drawable = gimp_drawable_get(param[2].data.d_drawable);

    switch (run_mode) {
        case GIMP_RUN_INTERACTIVE:
            /* Get options last values if needed */
            gimp_get_data("plug-in-multi-res-filter", &filter_vals);

            /* Display the dialog */
            if (!multi_res_dialog(drawable))
                return;
            break;

        case GIMP_RUN_NONINTERACTIVE:
            fprintf(stderr, "noninteractive\n");
            if (nparams != 6)
                status = GIMP_PDB_CALLING_ERROR;
            if (status == GIMP_PDB_SUCCESS)
                filter_vals.sigma_d = param[3].data.d_int32;
                filter_vals.sigma_r = param[4].data.d_int32;
                filter_vals.gpu = param[5].data.d_int32;
                filter_vals.use_float = param[6].data.d_int32;
                if (filter_vals.gpu) filter_vals.gpu = has_cuda_device();
            break;

        case GIMP_RUN_WITH_LAST_VALS:
            /*  Get options last values if needed  */
            gimp_get_data("plug-in-multi-res-filter", &filter_vals);
            break;

        default:
            break;
    }

    run_multi_filter(drawable, NULL);

    gimp_displays_flush();
    gimp_drawable_detach(drawable);

    /*  Finally, set options in the core  */
    if (run_mode == GIMP_RUN_INTERACTIVE)
      gimp_set_data("plug-in-multi-res-filter", &filter_vals, sizeof(filter_config));

    return;
}


void run_multi_filter(GimpDrawable *drawable, GimpPreview *preview) {
    gint         x1, y1, x2, y2;
    GimpPixelRgn rgn_in, rgn_out;
    guchar       *g0;
    gint         width, height, channels;

    if (!preview)
        gimp_progress_init("Multiresolution filter...");

    /* Gets upper left and lower right coordinates,
     * and layers number in the image */
    if (preview) {
        gimp_preview_get_position(preview, &x1, &y1);
        gimp_preview_get_size(preview, &width, &height);
        x2 = x1 + width;
        y2 = y1 + height;
    } else {
        gimp_drawable_mask_bounds(drawable->drawable_id, &x1, &y1, &x2, &y2);
        width = x2 - x1;
        height = y2 - y1;
    }

    channels = gimp_drawable_bpp(drawable->drawable_id);

    /* Initializes two PixelRgns, one to read original data,
     * and the other to write output data. That second one will
     * be merged at the end by the call to
     * gimp_drawable_merge_shadow() */
    gimp_pixel_rgn_init(&rgn_in,
                         drawable,
                         x1, y1,
                         width, height,
                         FALSE, FALSE);
    gimp_pixel_rgn_init(&rgn_out,
                         drawable,
                         x1, y1,
                         width, height,
                         preview == NULL, TRUE);

    /* Allocate memory for input and output image */
    g0 = g_new(guchar, width*height*channels);

    gimp_pixel_rgn_get_rect(&rgn_in, g0, x1, y1, width, height);

    /* Call GPU or CPU implementation */
    if (filter_vals.gpu) {
        run_gpu(g0, g0, width, height, channels, filter_vals.sigma_d, filter_vals.sigma_r, filter_vals.use_float);
    } else {
        run_cpu(g0, g0, width, height, channels, filter_vals.sigma_d, filter_vals.sigma_r, filter_vals.use_float);
    }

    gimp_pixel_rgn_set_rect(&rgn_out, g0, x1, y1, width, height);
    g_free(g0);

    /*  Update the modified region */
    if (preview) {
        gimp_drawable_preview_draw_region(GIMP_DRAWABLE_PREVIEW(preview), &rgn_out);
    } else {
        gimp_drawable_flush(drawable);
        gimp_drawable_merge_shadow(drawable->drawable_id, TRUE);
        gimp_drawable_update(drawable->drawable_id, x1, y1, width, height);
    }
}

