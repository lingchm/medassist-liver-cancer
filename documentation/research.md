# Liver Cancer 

Liver cancer is a significant global health concern, with more than 800,000 people diagnosed with this cancer each year worldwide and with primary liver cancer being a leading cause of cancer deaths worldwide (American Cancer Society, 2024). Liver is also a common destination for metastatic cancer cells originating from various abdominal organs, including the colon, rectum, pancreas, as well as distant organs such as the breast and lung. Consequently, meticulous examination of the liver and its lesions is integral to comprehensive tumor staging and management strategies. Standard protocols for tumor assessment, such as the Response Evaluation Criteria in Solid Tumor (RECIST), necessitate precise measurement of the diameter of the largest target lesion (Eisenhauer et al., 2009). Therefore, achieving accurate localization and precise segmentation of liver tumors within CT scans is imperative for the diagnosis, treatment planning, and monitoring of treatment response in patients with liver cancer (Shiina et al., 2018; Virdis et al., 2019).

## Deep Learning for Liver Tumor Segmentation  

Manual delineation of target lesions in CT scans is fraught with challenges, being both time-consuming and prone to poor reproducibility and operator-dependent variability. Automated liver tumor segmentation can provide clinicians with rapid and consistent tumor delineation. Recently, deep learning algorithms have showed promise for producing automated liver and tumor segmentation (Gul et al., 2002). While most algorithms
achieved exceptional performance in liver segmentation, with dice scores ranging from 0.90 to 0.96, enhancing liver tumor segmentation remains a challenge, currently standing at dice scores from 0.41 to 0.67 according to the recent Liver Tumor Segmentation Benchmark (LiTS) (Bilic et al.,2023).

## Challenges in Liver Tumor Segmentation

Liver tumor segmentation is an inherently challenging task. Tumors vary significantly in size, shape, and location across different patients, leading to a wide range of tumor characteristics and making it challenging for models to generalize (Sabir et al., 2022). Moreover, margins of some tumors are imprecise as CT scans exhibit low gentile brightness and roughness making it difficult to distinguish between tumor and healthy tissue (Sabir et al., 2022).

# Similar Tools on AI-assisted Segmentation

In recent years, AI-assisted segmentation has been trendy in research and industry.

A notable publicaly available online tool is [MedSeg](https://htmlsegmentation.s3.eu-north-1.amazonaws.com/index.html). This tool provides a number of pre-trained models to perform segmentation as well as annotate, edit, and save their segmentations. The tool has capability for organ segmentation of liver, lung, spleen, pancreas, and ventricles. 

Another great work is [MD.ai Annotator](https://md.ai/) that provides web-based annotation tools optimized for medical deep learning. It allows real-time collaboration within a team. The users can easily export the annotated images or prototype deep learning models using annotated data.

None of the existing tools has a focus on liver tumor segmentation, which is a very challenging task still demanding research for more accurate models.


# References

1.	American Cancer Society. (2024). Key Statistics About Liver Cancer. All about Cancer. https://www.cancer.org/cancer/types/liver-cancer/about/what-is-key-statistics.html
2.	Bilic, P., Christ, P., Li, H. B., Vorontsov, E., Ben-Cohen, A., Kaissis, G., Szeskin, A., Jacobs, C., Mamani, G. E. H., Chartrand, G., Lohöfer, F., Holch, J. W., Sommer, W., Hofmann, F., Hostettler, A., Lev-Cohain, N., Drozdzal, M., Amitai, M. M., Vivanti, R., … Menze, B. (2023). The Liver Tumor Seg-mentation Benchmark (LiTS). Medical Image Analysis, 84, 102680. https://doi.org/10.1016/j.media.2022.102680
3.	Eisenhauer, E. A., Therasse, P., Bogaerts, J., Schwartz, L. H., Sargent, D., Ford, R., Dancey, J., Arbuck, S., Gwyther, S., Mooney, M., Rubinstein, L., Shankar, L., Dodd, L., Kaplan, R., Lacombe, D., & Verweij, J. (2009). New response evaluation criteria in solid tumours: Revised RECIST guideline (version 1.1). European Journal of Cancer, 45(2), 228–247. https://doi.org/10.1016/j.ejca.2008.10.026
4.	Moawad, A. W., Fuentes, D., Morshid, A., Khalaf, A. M., Elmohr, M. M., Abusaif, A., Hazle, J. D., Kaseb, A. O., Hassan, M., Mahvash, A., Szklaruk, J., Qayyom, A., & Elsayes, K. (2021). Multimodality annotated HCC cases with and without advanced imaging segmentation (Version 1) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.5FNA-0924
5.	Moghbel, M., Mashohor, S., Mahmud, R., & Saripan, M. I. B. (2018). Review of liver segmentation and computer assisted detection/diagnosis methods in computed tomography. Artificial Intelligence Review, 50(4), 497–537. https://doi.org/10.1007/s10462-017-9550-x
6.	Shiina, S., Sato, K., Tateishi, R., Shimizu, M., Ohama, H., Hatanaka, T., Takawa, M., Nagamatsu, H., & Imai, Y. (2018). Percutaneous Ablation for Hepatocellular Carcinoma: Comparison of Various Ablation Techniques and Surgery. Canadian Journal of Gastroenterology and Hepatology, 2018, e4756147. https://doi.org/10.1155/2018/4756147
7.	Virdis, F., Reccia, I., Di Saverio, S., Tugnoli, G., Kwan, S. H., Kumar, J., Atzeni, J., & Podda, M. (2019). Clinical outcomes of primary arterial embo-lization in severe hepatic trauma: A systematic review. Diagnostic and Inter-ventional Imaging, 100(2), 65–75. https://doi.org/10.1016/j.diii.2018.10.004



