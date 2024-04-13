from PIL import Image, ImageDraw, ImageFilter

def add_shadow(image_path, output_path, opacity, direction, distance, softness):
    # Open the original image
    original = Image.open(image_path)

    # Create a new image that's big enough to fit the shadow
    width, height = original.size
    shadow_width = width + abs(distance) + softness * 2
    shadow_height = height + abs(distance) + softness * 2
    shadow = Image.new('RGBA', (shadow_width, shadow_height), (0, 0, 0, 0))

    # Draw the shadow rectangle, taking into account the direction and distance
    shadow_x = softness + max(distance, 0)
    shadow_y = softness + max(distance, 0)
    shadow_rectangle = [shadow_x, shadow_y, shadow_x + width, shadow_y + height]
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rectangle(shadow_rectangle, fill=(0, 0, 0, opacity))

    # Apply the Gaussian blur filter to the shadow
    shadow = shadow.filter(ImageFilter.GaussianBlur(softness * 0.35))

    # Paste the original image onto the shadow
    image_x = softness - min(distance, 0)
    image_y = softness - min(distance, 0)
    shadow.paste(original, (image_x, image_y))

    # Save the result
    shadow.save(output_path, 'PNG')