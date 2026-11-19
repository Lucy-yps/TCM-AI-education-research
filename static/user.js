/* 账户组件：登录状态 + 自动带 token + 顶部头像入口 */
(function(){
  var KT = "tcm_token", KU = "tcm_me";

  function tok(){ try { return localStorage.getItem(KT) || ""; } catch(e){ return ""; } }
  function me(){ try { return JSON.parse(localStorage.getItem(KU) || "null"); } catch(e){ return null; } }
  function setAuth(t, u){
    try {
      if (t) localStorage.setItem(KT, t); else localStorage.removeItem(KT);
      if (u) localStorage.setItem(KU, JSON.stringify(u)); else localStorage.removeItem(KU);
    } catch(e){}
  }

  /* 暴露给页面 */
  window.TCM = {
    token: tok,
    me: me,
    setAuth: setAuth,
    logout: function(){
      var t = tok();
      if (t) {
        fetch("/api/auth/logout", { method: "POST", headers: {"X-Token": t} }).catch(function(){});
      }
      setAuth(null, null);
      location.href = "/static/login.html";
    },
    requireLogin: function(){
      if (!tok()) { location.href = "/static/login.html?next=" + encodeURIComponent(location.pathname + location.search); return false; }
      return true;
    }
  };

  /* 拦截 fetch：加 X-Token（若有） */
  var _f = window.fetch;
  if (_f) {
    window.fetch = function(input, init){
      init = init || {};
      var t = tok();
      if (t) {
        try {
          var h = new Headers(init.headers || {});
          if (!h.has("X-Token")) h.set("X-Token", t);
          init.headers = h;
        } catch(e){}
      }
      return _f.call(window, input, init);
    };
  }

  /* 顶部挂头像 / 登录入口 */
  function mount(){
    var nav = document.querySelector(".nav")
           || document.querySelector(".nav-links")
           || document.querySelector(".topbar");
    if (!nav || document.getElementById("_uBox")) return;

    var box = document.createElement("a");
    box.id = "_uBox";
    box.href = "/static/account.html";
    box.style.cssText = "cursor:pointer;display:inline-flex;align-items:center;gap:5px;font-weight:600;margin-left:4px;text-decoration:none;color:#2F6B52;";

    function refresh(){
      var u = me();
      if (u) {
        box.textContent = (u.avatar || "👤") + " " + (u.display || u.username || "");
        box.title = "个人信息";
        box.href = "/static/account.html";
      } else {
        box.textContent = "👤 登录";
        box.title = "登录 / 注册";
        box.href = "/static/login.html";
      }
    }
    refresh();

    /* 每 60 秒静默刷新一次用户信息（昵称/头像可能改了） */
    if (tok()) {
      setTimeout(function(){
        fetch("/api/auth/me").then(function(r){ return r.ok ? r.json() : null; }).then(function(j){
          if (j && j.data) { setAuth(tok(), j.data); refresh(); }
        }).catch(function(){});
      }, 500);
    }

    nav.appendChild(box);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
