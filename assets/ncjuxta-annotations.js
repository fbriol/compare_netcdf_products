// Ctrl+S / Cmd+S clicks the save button of the visible annotation form.
document.addEventListener('keydown', function (event) {
    if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== 's') {
        return;
    }
    var buttons = document.querySelectorAll('[data-ctrl-s]');
    for (var i = 0; i < buttons.length; i++) {
        if (buttons[i].offsetParent !== null) {
            event.preventDefault();
            buttons[i].click();
            return;
        }
    }
});
