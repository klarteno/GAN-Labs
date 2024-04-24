import torch



def compute_average_faces(
    feature_idx, image_dim, data_loader, device=None, encoding_fn=None
):

    avg_img_with_feat = torch.zeros(image_dim, dtype=torch.float32)
    avg_img_without_feat = torch.zeros(image_dim, dtype=torch.float32)

    num_img_with_feat = 0
    num_images_without_feat = 0

    for images, labels in data_loader:
        idx_img_with_feat = labels[:, feature_idx].to(torch.bool)

        if encoding_fn is None:
            embeddings = images
        else:
            ####################################
            ### Get latent representation
            with torch.no_grad():

                if device is not None:
                    images = images.to(device)
                embeddings = encoding_fn(images).to("cpu")
            ####################################

        avg_img_with_feat += torch.sum(embeddings[idx_img_with_feat], axis=0)
        avg_img_without_feat += torch.sum(embeddings[~idx_img_with_feat], axis=0)
        num_img_with_feat += idx_img_with_feat.sum(axis=0)
        num_images_without_feat += (~idx_img_with_feat).sum(axis=0)

    avg_img_with_feat /= num_img_with_feat
    avg_img_without_feat /= num_images_without_feat

    return avg_img_with_feat, avg_img_without_feat
