import replicate

input = {
    "seed": 22,
    "image": "https://live.staticflickr.com/65535/54289737342_3411cbf596_n.jpg",
    "prompt": "an illustration of an apple",
    "structure": "canny",
    "image_resolution": 512,
    "eta": 0
}

output = replicate.run(
    "rossjillian/controlnet:795433b19458d0f4fa172a7ccf93178d2adb1cb8ab2ad6c8fdc33fdbcd49f477",
    input=input
)
for index, item in enumerate(output):
    with open(f"output_{index}.png", "wb") as file:
        file.write(item.read())
#=> output_0.png written to disk