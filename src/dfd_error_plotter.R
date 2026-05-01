### Import stuff
library(ggplot2)
library(dplyr)
library(stringr)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
save_plot = FALSE

import_cur_df <- function(pipeline_prefix, dist_prefix) {
  file_path = paste0("reports/MAE_outputs/",pipeline_prefix, "_fineAngleIncrement_ctcMAG_HAPYParamsWLensPitDistance-avgTierIdists_", dist_prefix, "Focus_angular_MAE_method_Habitat_and_textureImages_avgEyeTube.csv")
  return(read.csv(file_path,header=TRUE))
}

### 2 surface dfs
# wo Curvature
i = "FocalSplit-1dfComparisons-"
k = "inf"
woCurvature_TwoSurface_inf = import_cur_df(i,k)
k = "30cm"
woCurvature_TwoSurface_30cm = import_cur_df(i,k)
k = "10cm"
woCurvature_TwoSurface_10cm = import_cur_df(i,k)

# w Curvature
i = "FocalSplit-1dfComparisons-wFieldCurvature"
k = "inf"
TwoSurface_inf = import_cur_df(i,k)
k = "30cm"
TwoSurface_30cm = import_cur_df(i,k)
k = "10cm"
TwoSurface_10cm = import_cur_df(i,k)

candidate_datasets = list('FS_2surface_noCurvature_inf' = woCurvature_TwoSurface_inf, 'FS_2surface_noCurvature_30cm' = woCurvature_TwoSurface_30cm, 'FS_2surface_noCurvature_10cm' = woCurvature_TwoSurface_10cm,
                          'FS_2surface_inf' = TwoSurface_inf, 'FS_2surface_30cm' = TwoSurface_30cm, 'FS_2surface_10cm' = TwoSurface_10cm)


ggplot(data=candidate_datasets[["FS_2surface_noCurvature_inf"]],aes(x=Ground_depth.m.,y=MAE.m.))+
  #geom_line(color='green4') +
  #geom_point(color='green4') +
  geom_smooth(se=FALSE,color='green4',linewidth=1.5) +
  geom_line(data=candidate_datasets[["FS_2surface_inf"]],aes(x=Ground_depth.m.,y=MAE.m.), color='black',linewidth=1.5) +
  #geom_point(data=candidate_datasets[["FS_2surface_inf"]],aes(x=Ground_depth.m.,y=MAE.m.), color='orange') +
  geom_abline(slope=0.1,intercept=0,linetype='dashed',color='blue',linewidth=0.8) +
  scale_y_continuous(limits=c(0,16),expand=c(0,0)) +
  scale_x_continuous(limits=c(0,3.2),expand=c(0,0)) +
  labs(y='Depth estimation error (m)', x="Ground depth (m)") + theme(legend.position="bottom") +
  theme_classic() + theme(axis.title=element_text(size=18),axis.text=element_text(size=12),panel.grid.minor = element_blank(),panel.grid.major = element_blank(),legend.position="none")# + coord_flip()
if (save_plot) {ggsave(paste0('DFD_farErrors_CurvatureVSnoCurvature.png'),width=1500,height=1500,units='px')}

ggplot(data=candidate_datasets[["FS_2surface_inf"]],aes(x=Ground_depth.m.*1000,y=MAE.m.*1000))+
  geom_line(color='black', linewidth=1.5) +
  #geom_point(color='orange') +
  geom_line(data=candidate_datasets[["FS_2surface_30cm"]],aes(x=Ground_depth.m.*1000,y=MAE.m.*1000), color='darkorange2',alpha=0.7, linewidth=1.5) +
  #geom_point(data=candidate_datasets[["FS_2surface_30cm"]],aes(x=Ground_depth.m.*100,y=MAE.m.*1000), color='darkorange2') +
  geom_line(data=candidate_datasets[["FS_2surface_10cm"]],aes(x=Ground_depth.m.*1000,y=MAE.m.*1000), color='red',alpha=0.5, linewidth=1.5) +
  #geom_point(data=candidate_datasets[["FS_2surface_10cm"]],aes(x=Ground_depth.m.*100,y=MAE.m.*1000), color='red') +
  #geom_abline(slope=1,intercept=0,linetype='dashed') +
  geom_abline(slope=0.1,intercept=0,linetype='dashed',color='blue',linewidth=1.2) +
  #geom_hline(yintercept = 3,color='black') +
  geom_vline(xintercept = 7,color='grey') +
  scale_y_continuous(limits=c(0,49),expand=c(0,0)) +
  scale_x_continuous(limits=c(0,100),expand=c(0,0)) +
  labs(y='Depth estimation error (mm)', x="Ground depth (mm)") + theme(legend.position="bottom") +
  theme_classic() + theme(axis.title=element_text(size=18),axis.text=element_text(size=12),panel.grid.minor = element_blank(),panel.grid.major = element_blank(),legend.position="none")# + coord_flip()
if (save_plot) {ggsave(paste0('DFD_closeupErrors_MultiDist.png'),width=1500,height=1500,units='px')}

print('The multiple "roots" can be explained looking at the optical power needed to bring the ground in to focus, and looking where this line coincides with the optical power of the lens under incidence.')
print('Orange intersects the line twice, once at infinity, once at ~1 cm (9deg)')
print('DarkOrange2 intersects the line twice, once at ~23cm (~0.7deg), once at ~1 (~8deg)')
print('Red intersects the line twice, once at ~6.5cm (~3.5deg), once at ~1.5cm (~4.5deg)')
require(magick)
plot(image_read("reports/demo/OpticalPowers.png"))


