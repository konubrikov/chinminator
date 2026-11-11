# Chinminator 1.7.0 — preview

## Русский
Локальная ретушь подбородка в Photoshop. Qwen-Image 2.1, FLUX 3, Lightning 4-step и Pruna 6-step v0.2; выбор модели при запуске, загрузка недостающих компонентов, переключение через Choose-model. Поддерживаются Photoshop Actions и Обработчик изображений. Результат добавляется новым слоем; фотографии обрабатываются локально.

Скачайте ZIP своей платформы, распакуйте в постоянную папку, прочитайте START-HERE.txt, установите Chinminator.ccx через Adobe и запустите Start. Веса моделей не включены и скачиваются при установке. Панель CCX также приложена отдельно.

Windows: 10/11 x64, NVIDIA Turing+, минимум 8 ГБ VRAM / 16 ГБ RAM (32 ГБ рекомендуется), драйвер с поддержкой CUDA 13. macOS: 14+, Apple Silicon, минимум 24 ГБ unified memory. Диск: от 45 ГБ, для BF16 — 80 ГБ. Photoshop 24.2+, для Actions — 25+, RGB 8 бит/канал.

Ограничения: preview; качество эвристик и результат ретуши требуют проверки пользователем. Для FLUX 3 нужен собственный доступ Hugging Face и принятие условий модели. Использование моделей регулируется их отдельными лицензиями. Пакеты не подписаны. Реальная генерация на macOS/MPS и установка на чистой ОС не проверены. В этом проходе выполнены 34 контрактных теста сервера, 13 тестов профилей, Photoshop Actions contracts и проверка структуры архивов/CCX; это не проверка реальной генерации всех четырёх вариантов. Старые validation-файлы в исходниках относятся к указанным в них версиям.

## English
Local chin retouching for Photoshop. Includes startup selection for Qwen-Image 2.1, FLUX 3, Lightning 4-step and Pruna 6-step v0.2, dependency setup, Choose-model, Photoshop Actions and Image Processor integration. Results are added as a new layer. Photos are processed locally.

Download the ZIP for your platform, extract to a permanent folder, read START-HERE.txt, install Chinminator.ccx through Adobe and run Start. Model weights are downloaded during setup and are not bundled. The CCX panel is also attached separately. Current setup instructions are in Russian.

Windows 10/11 x64, NVIDIA Turing+, 8 GB VRAM / 16 GB RAM minimum (32 GB recommended), CUDA 13-capable driver. macOS 14+, Apple Silicon, at least 24 GB unified memory. Free disk: 45 GB minimum, 80 GB for BF16. Photoshop 24.2+; Actions require 25+; RGB 8-bit documents.

Preview limitations: heuristic detection and output quality require user review. FLUX 3 requires your own Hugging Face access and acceptance of its model terms. Model licenses apply separately. Packages are unsigned. Actual macOS/MPS generation and clean-OS installation have not been verified. This preparation passed 34 server contract tests, 13 profile tests, Actions contracts, archive/CCX validation; it does not certify actual inference for all model variants. Earlier validation files retain their original version scope.
