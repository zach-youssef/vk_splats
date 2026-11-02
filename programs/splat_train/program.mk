# test.sh renders the scene's ground truth with splat_render (make_synthetic.py render-gt), native
# or for the web as the test is
test-splat_train: $(OUT)/splat_render/splat_render
test-web-splat_train: $(OUT)/web/splat_render
