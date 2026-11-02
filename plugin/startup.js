/* Вызывается Photoshop при загрузке плагина (manifest.executioncontext.startup).
   Регистрируем команду меню, открывающую панель. */
const { app } = require("uxp").shell ? require("uxp") : { app: null };
module.exports = () => {
  // Панель открывается автоматически из Plugins > Plugins > Double Chin Fix.
  // Здесь можно зарегистрировать hot-key/команду — оставлено минимальным.
};
