### Хакатон "Лидеры цифровой трансформации" 2024 (7 место из 70 команд)
#### Кейс: Предиктивная модель для рекомендации продуктов банка (Задача №12)  
##### Кейсодержатель: Сбер
https://i.moscow/lct  
https://vk.com/leaders_hack?offset=20&own=1  

***
### Презентация проекта
[Итоговая презентация решения на защите](https://docs.google.com/presentation/d/1scHjXJTQUxrV4VAHMru-RDn-KhWszgVZ/edit?usp=sharing&ouid=113491937784577068477&rtpof=true&sd=true)


### Структура репозитария:
* baseline_v1_8_7.ipynb - код обучения 
* calc_good_bad_cos_sim_v2.ipynb - дополнительный расчет фичей
* compression_data.py - предобработка данных, сжатие для дальнейшей работы
* create_agg_dialog_v7_2.ipynb - код по генерации фичей по диалогам
* create_agg_geo_v7_1 - код по генерации фичей по диалогам
* create_agg_ptls_v7_1.ipynb - код по генерации векторов pytorch-lifestream
* create_agg_target_v2.ipynb - код по генерации фичей на базе таргетов
* create_agg_trx_v7_1 - код по генерации фичей на базе транзакций
* create_geo_features.py - дополнительный код по генерации ГЕО фичей
* create_normalize_trx.ipynb - дополнительный код по обработке транзакций
* gen_sample_pairs_Client_Month.ipynb - генератор сэмплов
* list_uniq_client_target.csv - фичи по клиентам  
* client_without_dlg.csv - фичи по диалогам клиентов
* client_without_geo.csv - ГЕО фичи 
* client_without_trx.csv - фичи по транзакциям
* Описание решения.docx - Описание решения 
* Фичи ЛЦТ Сбер 2024.xlsx - описание фичей
* Анализ экспериментов.xlsx - Результаты экспериментов

### Документация
[Описание решения](https://docs.google.com/document/d/1BLEOxWLion65820VTbzDnzD_MNWplLjiPMyXjs5G6Ko/edit?usp=sharing)  
[Описание фичей](https://docs.google.com/spreadsheets/d/1k-VmFNZEMRYDlroP6Bss-BNT-ZUdGs2ygX_Jym4cNqI/edit?usp=sharing)  
[Результаты экспериментов](https://docs.google.com/spreadsheets/d/1UEw8heslMy-hNOawX51f8UvLDqi5eMlRmSSIw45g0Fk/edit?usp=sharing)
