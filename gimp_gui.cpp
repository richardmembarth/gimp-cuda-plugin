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

#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>

#include "gimp_main.hpp"
#include "gimp_gui.hpp"
#include "multi_res_cpu.hpp"
#include "multi_res_host.hpp"
#include "defines_cpu.hpp"


gboolean multi_res_dialog (GimpDrawable *drawable) {
    GtkWidget *dialog;
    GtkWidget *main_vbox;
    GtkWidget *main_hbox;
    GtkWidget *preview;
    GtkWidget *gpu_button;
    GtkWidget *float_button;
    GtkWidget *frame;
    GtkWidget *spread_d_label;
    GtkWidget *spread_r_label;
    GtkWidget *alignment;
    GtkWidget *spinbutton_d;
    GtkWidget *spinbutton_r;
    GtkObject *spinbutton_adj_d;
    GtkObject *spinbutton_adj_r;
    GtkWidget *frame_label;
    gboolean   run;
    
    gimp_ui_init("Multiresolution filter", FALSE);
    
    dialog = gimp_dialog_new("Multiresolution filter", "Multiresolution filter",
                              NULL, (GtkDialogFlags) 0,
                              gimp_standard_help_func, "plug-in-multi-res-filter",
                              GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
                              GTK_STOCK_OK,     GTK_RESPONSE_OK,
                              NULL);
    
    main_vbox = gtk_vbox_new(FALSE, 6);
    gtk_container_add(GTK_CONTAINER(GTK_DIALOG (dialog)->vbox), main_vbox);
    gtk_widget_show(main_vbox);
    
    preview = gimp_drawable_preview_new(drawable, &filter_vals.preview);
    gtk_box_pack_start(GTK_BOX(main_vbox), preview, TRUE, TRUE, 0);
    gtk_widget_show(preview);
    
    gpu_button = gtk_check_button_new_with_mnemonic("_Use GPU");
    gtk_box_pack_start(GTK_BOX(main_vbox), gpu_button, FALSE, FALSE, 0);
    if (has_cuda_device()) {
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gpu_button), TRUE);
        gtk_widget_set_sensitive(gpu_button, TRUE);
        filter_vals.gpu = TRUE;
    } else {
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gpu_button), FALSE);
        gtk_widget_set_sensitive(gpu_button, FALSE);
        filter_vals.gpu = FALSE;
    }
    g_signal_connect(gpu_button, "toggled",
                              G_CALLBACK(gimp_toggle_button_update),
                              &filter_vals.gpu);
    g_signal_connect_swapped(gpu_button, "toggled",
                              G_CALLBACK(gimp_preview_invalidate),
                              preview);
    gtk_widget_show(gpu_button);
    
    float_button = gtk_check_button_new_with_mnemonic("Use more accurate calculation (_float)");
    gtk_box_pack_start(GTK_BOX(main_vbox), float_button, FALSE, FALSE, 0);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(float_button), FALSE);
    filter_vals.use_float = FALSE;
    g_signal_connect(float_button, "toggled",
                              G_CALLBACK(gimp_toggle_button_update),
                              &filter_vals.use_float);
    g_signal_connect_swapped(float_button, "toggled",
                              G_CALLBACK(gimp_preview_invalidate),
                              preview);
    gtk_widget_show(float_button);

    frame = gtk_frame_new(NULL);
    gtk_widget_show(frame);
    gtk_box_pack_start(GTK_BOX(main_vbox), frame, TRUE, TRUE, 0);
    gtk_container_set_border_width(GTK_CONTAINER(frame), 6);
    
    alignment = gtk_alignment_new(0.5, 0.5, 1, 1);
    gtk_widget_show(alignment);
    gtk_container_add(GTK_CONTAINER(frame), alignment);
    gtk_alignment_set_padding(GTK_ALIGNMENT(alignment), 6, 6, 6, 6);
    
    main_hbox = gtk_hbox_new(FALSE, 0);
    gtk_widget_show(main_hbox);
    gtk_container_add(GTK_CONTAINER(alignment), main_hbox);
    
    spread_d_label = gtk_label_new_with_mnemonic("_Geometric spread:");
    gtk_widget_show(spread_d_label);
    gtk_box_pack_start(GTK_BOX (main_hbox), spread_d_label, FALSE, FALSE, 6);
    gtk_label_set_justify(GTK_LABEL (spread_d_label), GTK_JUSTIFY_RIGHT);
    
    spinbutton_d = gimp_spin_button_new(&spinbutton_adj_d, filter_vals.sigma_d,
                                       1, MAX_SIGMA_D, 1, 1, 0, (MAX_SIGMA_D>3)?3:MAX_SIGMA_D, 0);
    gtk_box_pack_start(GTK_BOX(main_hbox), spinbutton_d, FALSE, FALSE, 0);
    gtk_widget_show(spinbutton_d);
    
    spread_r_label = gtk_label_new_with_mnemonic("_Photometric spread:");
    gtk_widget_show(spread_r_label);
    gtk_box_pack_start(GTK_BOX(main_hbox), spread_r_label, FALSE, FALSE, 6);
    gtk_label_set_justify(GTK_LABEL(spread_r_label), GTK_JUSTIFY_RIGHT);
    
    spinbutton_r = gimp_spin_button_new(&spinbutton_adj_r, filter_vals.sigma_r,
                                       1, MAX_SIGMA_R, 1, 1, 0, (MAX_SIGMA_R>3)?3:MAX_SIGMA_R, 0);
    gtk_box_pack_start(GTK_BOX(main_hbox), spinbutton_r, FALSE, FALSE, 0);
    gtk_widget_show(spinbutton_r);
    
    frame_label = gtk_label_new("<b>Modify spreads</b>");
    gtk_widget_show(frame_label);
    gtk_frame_set_label_widget(GTK_FRAME(frame), frame_label);
    gtk_label_set_use_markup(GTK_LABEL(frame_label), TRUE);
    
    g_signal_connect_swapped(preview, "invalidated",
                              G_CALLBACK(run_multi_filter),
                              drawable);
    g_signal_connect_swapped(spinbutton_adj_d, "value_changed",
                              G_CALLBACK(gimp_preview_invalidate),
                              preview);
    
    g_signal_connect_swapped(spinbutton_adj_r, "value_changed",
                              G_CALLBACK(gimp_preview_invalidate),
                              preview);
    
    g_signal_connect(spinbutton_adj_d, "value_changed",
                      G_CALLBACK(gimp_int_adjustment_update),
                      &filter_vals.sigma_d);
    g_signal_connect(spinbutton_adj_r, "value_changed",
                      G_CALLBACK(gimp_int_adjustment_update),
                      &filter_vals.sigma_r);

    gtk_widget_show(dialog);
    
    run_multi_filter(drawable, GIMP_PREVIEW(preview));
    
    run = (gimp_dialog_run(GIMP_DIALOG(dialog)) == GTK_RESPONSE_OK);
    
    gtk_widget_destroy(dialog);
    
    return run;
}

