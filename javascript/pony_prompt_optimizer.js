(function () {
    const PUSH_BUTTON_ID = "ppo-push-button";
    const PREVIEW_ID = "ppo-preview";

    const getApp = () => (window.gradioApp && window.gradioApp()) || document.querySelector("gradio-app");

    function readPreview() {
        const app = getApp();
        if (!app) return "";
        const textarea = app.querySelector(`#${PREVIEW_ID} textarea`);
        return textarea ? textarea.value : "";
    }

    function updatePromptAreas(value) {
        if (!value) return;
        const app = getApp();
        if (!app) return;
        const selectors = ["#txt2img_prompt textarea", "#img2img_prompt textarea"];
        selectors.forEach((selector) => {
            const area = app.querySelector(selector);
            if (!area) return;
            area.value = value;
            area.dispatchEvent(new Event("input", { bubbles: true }));
            area.dispatchEvent(new Event("change", { bubbles: true }));
        });
    }

    function attach() {
        const app = getApp();
        if (!app) return;
        const button = app.querySelector(`#${PUSH_BUTTON_ID} button, #${PUSH_BUTTON_ID}`);
        if (!button || button.dataset.ppoBound === "true") return;
        button.dataset.ppoBound = "true";
        button.addEventListener("click", () => {
            const preview = readPreview();
            if (!preview) return;
            updatePromptAreas(preview);
        });
    }

    const observer = new MutationObserver(() => attach());
    observer.observe(document.documentElement, { childList: true, subtree: true });
    attach();
})();
