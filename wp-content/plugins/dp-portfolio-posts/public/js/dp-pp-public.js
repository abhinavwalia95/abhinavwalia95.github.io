jQuery(document).ready(function () {

    jQuery(window).load(function () {
        jQuery('.dp_ppp_module.et_pb_filterable_portfolio').each(function () {
            var active_filter = jQuery(this).attr('data-filter');
            if (active_filter !== "") {
                var parent = jQuery(this);
                var filter = parent.find('.ppp_filterable_link[data-category-filter="' + active_filter + '"]');
                filter.parent('li').siblings('li.et_pb_portfolio_filter_all').children('a').removeClass('active');
                setTimeout(function () {
                    filter.click();
                }, '1');
            }
        });
    });

    jQuery(document).on('click', '.et_pb_portfolio_filter', function () {
        var items_container = jQuery(this).parent('ul').parent('div').siblings('div.et_pb_portfolio_items_wrapper').children('div.et_pb_portfolio_items');
        items_container.children('div.et_pb_portfolio_item').children('a.et_pb_lightbox_image').each(function () {
            jQuery(this).magnificPopup({
                type: 'image',
                removalDelay: 500,
                mainClass: 'mfp-fade',
                zoom: {
                    enabled: true,
                    duration: 500,
                    opener: function (element) {
                        return element.find('img');
                    }
                }
            });
        });
    });

    jQuery('#modal_inner iframe').load(function () {
        jQuery('.et_pb_loader_img').css('display', 'none');
        jQuery(this).closest('.modal').addClass('active');
    });

    jQuery(document).on('click', '.et_pb_lightbox_post_popup', function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        jQuery('body').css({"overflow-y": 'hidden'});
        var iphone = ['iPhone'].indexOf(navigator.platform) >= 0;
        var ipad = ['iPad'].indexOf(navigator.platform) >= 0;
        jQuery('iframe#dp_iframe').attr("src", jQuery(this).attr("data-ajaxurl"));
        jQuery('.et_pb_loader_img').css('display', 'block');
        var section = jQuery("body");
        var modal_overlay = jQuery("body .modal_overlay");
        var modal = jQuery("body .modal_overlay .modal");
        section.append(modal_overlay);
        modal_overlay.css('display', 'block');
        modal.css('display', 'block');
        if (iphone || ipad) {
            jQuery('.modal_inner').addClass('ios');
        }
    });

    jQuery(document).on('click', '.modal_overlay, .pop_up_close_btn', function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        jQuery('iframe#dp_iframe').contents().empty();
        jQuery('body .modal_overlay').css('display', 'none');
        jQuery('.et_pb_loader_img').css('display', 'block');
        jQuery(this).closest('.modal').removeClass('active');
        jQuery('body').css({"overflow-y": 'scroll'});
    });

    jQuery(document).on('click', '.dp_ppp_module a.dp_ppp_lightbox_image', function (ev) {
        ev.preventDefault();
        var args = {
            type: 'image',
            removalDelay: 700,
            mainClass: 'mfp-fade dp-ppp-lightbox'
        };
        var dp_ppp_gallery_images = [];
        var dp_ppp_gallery_titles = [];
        var index = 0;
        var start_at = 0;
        var trigger_by = jQuery(this).prop('href');
        if (jQuery(this).find('.dp_ppp_post_thumb').attr('data-lightbox-gallery') === 'on') {
            jQuery(this).parents('.et_pb_module').find('.dp_ppp_lightbox_image').each(function () {
                if (jQuery(this).find('.dp_ppp_post_thumb').length !== 0) {
                    if (trigger_by === jQuery(this).prop('href')) {
                        start_at = index;
                    }
                    index++;
                    dp_ppp_gallery_images.push(jQuery(this).prop('href'));
                    dp_ppp_gallery_titles.push(jQuery(this).find('.dp_ppp_post_thumb').attr('alt'));
                }
            });
            args['gallery'] = {
                enabled: true,
                navigateByImgClick: true
            };
        } else {
            if (jQuery(this).find('.dp_ppp_post_thumb').length !== 0) {
                dp_ppp_gallery_images.push(jQuery(this).prop('href'));
                dp_ppp_gallery_titles.push(jQuery(this).find('.dp_ppp_post_thumb').attr('alt'));
            }
        }
        for (var item in dp_ppp_gallery_images) {
            dp_ppp_gallery_images[item] = {'src': dp_ppp_gallery_images[item], 'title': dp_ppp_gallery_titles[item]};
        }
        args['items'] = dp_ppp_gallery_images;
        jQuery.magnificPopup.open(args, start_at);
    });
});
