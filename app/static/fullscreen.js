// Full-screen toggle for Plotly figures.
// A ".ncj-fullscreen-btn" button inside a ".ncj-plot-container" element
// toggles that container in full screen; the figure is resized to the
// screen size and restored on exit (Esc). Shared by the Dash app and
// the standalone annotated report.
(function () {
    'use strict';

    function graphOf(container) {
        return container.querySelector('.js-plotly-plot');
    }

    function targetHeight(container) {
        var header = container.firstElementChild;
        var used = header ? header.offsetHeight : 0;
        return Math.max(300, window.innerHeight - used - 32);
    }

    function targetWidth(container) {
        var style = window.getComputedStyle(container);
        var padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
        return Math.max(300, container.clientWidth - padding);
    }

    function fit(container) {
        var gd = graphOf(container);
        if (!gd || !window.Plotly || !gd.layout) return;
        var size = {height: targetHeight(container), width: targetWidth(container)};
        if (gd.layout.height !== size.height || gd.layout.width !== size.width) {
            window.Plotly.relayout(gd, size);
        }
    }

    function enter(container) {
        var gd = graphOf(container);
        if (!gd || !window.Plotly || !gd.layout) return;
        if (!('ncjHeight' in gd.dataset)) {
            gd.dataset.ncjHeight = gd.layout.height || '';
            gd.dataset.ncjWidth = gd.layout.width || '';
        }
        if (!gd.ncjAfterPlot && gd.on) {
            // Server callbacks (zoom resampling) replace the figure and
            // its height: keep it fitted while in full screen.
            gd.ncjAfterPlot = true;
            gd.on('plotly_afterplot', function () {
                if (document.fullscreenElement === container) fit(container);
            });
        }
        fit(container);
        window.dispatchEvent(new Event('resize'));
    }

    function leave(container) {
        var gd = graphOf(container);
        if (!gd || !window.Plotly || !('ncjHeight' in gd.dataset)) return;
        var height = gd.dataset.ncjHeight;
        var width = gd.dataset.ncjWidth;
        delete gd.dataset.ncjHeight;
        delete gd.dataset.ncjWidth;
        window.Plotly.relayout(gd, {
            height: height ? Number(height) : null,
            width: width ? Number(width) : null,
        });
        window.dispatchEvent(new Event('resize'));
    }

    document.addEventListener('click', function (event) {
        var button = event.target.closest && event.target.closest('.ncj-fullscreen-btn');
        if (!button) return;
        var container = button.closest('.ncj-plot-container');
        if (!container) return;
        event.preventDefault();
        if (document.fullscreenElement === container) {
            document.exitFullscreen();
        } else if (container.requestFullscreen) {
            container.requestFullscreen();
        }
    });

    var current = null;
    document.addEventListener('fullscreenchange', function () {
        var container = document.fullscreenElement;
        if (current && current !== container) leave(current);
        current = container && container.classList.contains('ncj-plot-container')
            ? container : null;
        if (current) enter(current);
    });
})();
