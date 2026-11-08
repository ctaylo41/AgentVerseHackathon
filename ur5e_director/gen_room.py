"""Offline Multimodal Room Collision Generator for Antigravity (`gen_room.py`).

This standalone setup utility reads a real-world photograph of the filming
workspace (`room.jpg`), compresses it in memory, and sends it to the Google
Gemini multimodal API (`gemini-4-argon` / `gemini-3.8-flash`) to synthesize a
MuJoCo collision geometry file (`generated_room.xml`).

Role in System Architecture:
  - Run once offline before starting the simulation if the physical studio
    layout changes.
  - Produces `generated_room.xml`, which is statically `<include>`d by the
    master `scene.xml` file so the UR5e robot arm respects real-world walls,
    desks, and the designated `creator_zone` floor mat.
  - Includes an automatic deterministic fallback room layout if the API is
    unreachable or returns non-XML output.
"""

import base64
import os
import re
from typing import Optional, Tuple
import cv2
from dotenv import load_dotenv
from google import genai

load_dotenv()


def compress_room_image(
    image_path: str = "room.jpg", max_width: int = 640, jpeg_quality: int = 80
) -> Tuple[str, bytes]:
    """Loads, resizes, and JPEG-compresses the workspace photo for multimodal prompting.

    Args:
        image_path: Path to the source workspace photograph. Defaults to
            `'room.jpg'`.
        max_width: Maximum width in pixels after aspect-preserving resize.
            Defaults to `640`.
        jpeg_quality: OpenCV JPEG compression quality (`0` to `100`). Defaults
            to `80`.

    Returns:
        A tuple `(b64_image, raw_bytes)` where `b64_image` is the UTF-8 base64
        encoded string and `raw_bytes` is the compressed JPEG byte buffer.

    Raises:
        FileNotFoundError: If `image_path` does not exist or cannot be decoded.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(
            f"Could not read {image_path}! Make sure it is a valid image in this folder."
        )

    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * (max_width / w))))
    _, buffer = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    b64_image = base64.b64encode(buffer).decode("utf-8")
    return b64_image, buffer.tobytes()


def generate_room_xml(
    image_path: str = "room.jpg", output_xml_path: str = "generated_room.xml"
) -> str:
    """Generates `generated_room.xml` from `room.jpg` via Gemini multimodal inference.

    Attempts the Google GenAI Interactions API first, then `models.generate_content`,
    and finally falls back to a safe default studio geometry if offline.

    Args:
        image_path: Input workspace image path. Defaults to `'room.jpg'`.
        output_xml_path: Destination path for the generated MuJoCo XML fragment.
            Defaults to `'generated_room.xml'`.

    Returns:
        The final XML string written to `output_xml_path`.

    Raises:
        ValueError: If `GEMINI_API_KEY` is missing from the environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env!")
    print(f"[1/3] Loaded API Key ({api_key[:6]}...)", flush=True)

    client = genai.Client(http_options={"timeout": 15000})

    print(f"[2/3] Compressing {image_path} in memory...", flush=True)
    b64_image, raw_jpeg_bytes = compress_room_image(image_path)

    prompt = """
Look at this workspace photo and write a valid MuJoCo XML file that models the room surfaces 
(walls and a desk/table off to the side) as basic collision boxes (geoms) so a robot arm at (0, 0, 0) doesn't hit them.
Keep the area around x=[0.4, 2.0], y=[-1.0, 1.0] clear for a person to walk.
Include a green translucent floor marker geom named "creator_zone" at pos="1.2 0 0.002" size="0.8 1.0 0.001" rgba="0.1 0.8 0.2 0.3" contype="0" conaffinity="0".
IMPORTANT: Wrap the entire output in <mujoco><worldbody> ... </worldbody></mujoco> tags.
Do NOT include a floor plane. Output ONLY raw XML.
"""

    xml_output = ""
    used_model: Optional[str] = None

    for model_id in ["gemini-4-argon", "gemini-3.8-flash"]:
        print(f"[3/3] Requesting XML from {model_id} (15s timeout)...", flush=True)
        try:
            interaction = client.interactions.create(
                model=model_id,
                input=[
                    {"type": "image", "data": b64_image, "mime_type": "image/jpeg"},
                    {"type": "text", "text": prompt},
                ],
                generation_config={"thinking_level": "low"},
            )
            raw_text = (interaction.output_text or "").strip()
            if raw_text:
                xml_output = raw_text
                used_model = model_id
                break
        except Exception as exc:
            print(
                f"      {model_id} interactions API failed ({type(exc).__name__}: {exc})",
                flush=True,
            )
            try:
                from google.genai import types

                response = client.models.generate_content(
                    model=model_id,
                    contents=[
                        types.Part.from_bytes(
                            data=raw_jpeg_bytes, mime_type="image/jpeg"
                        ),
                        prompt,
                    ],
                )
                raw_text = (response.text or "").strip()
                if raw_text:
                    xml_output = raw_text
                    used_model = f"{model_id} (generate_content)"
                    break
            except Exception as exc2:
                print(
                    f"      {model_id} generate_content fallback failed ({type(exc2).__name__}: {exc2})",
                    flush=True,
                )

    # Extract only valid <mujoco>...</mujoco> or <worldbody>...</worldbody> block.
    match = re.search(
        r"(<mujoco>[\s\S]*?</mujoco>|<worldbody>[\s\S]*?</worldbody>)", xml_output
    )
    if match:
        xml_output = match.group(1)
        if xml_output.startswith("<worldbody>"):
            xml_output = f"<mujoco>\n{xml_output}\n</mujoco>"
    else:
        xml_output = ""

    print("\n" + "=" * 60)
    if xml_output.strip():
        room_status = f"CUSTOM ROOM (Generated from {image_path} via {used_model})"
        print(f"SUCCESS: Using {room_status}")
        xml_output = f"<!-- ROOM_SOURCE: {room_status} -->\n" + xml_output
    else:
        room_status = "DEFAULT FALLBACK ROOM (API failed or returned invalid XML)"
        print(f"WARNING: Using {room_status}")
        xml_output = f"""<!-- ROOM_SOURCE: {room_status} -->
<mujoco>
    <worldbody>
        <!-- Back and side walls -->
        <geom name="back_wall" type="box" pos="2.3 0 1.0" size="0.05 2.0 1.0" rgba="0.85 0.85 0.88 1"/>
        <geom name="left_wall" type="box" pos="1.0 2.0 1.0" size="1.3 0.05 1.0" rgba="0.82 0.82 0.85 1"/>
        <!-- Studio desk off to the right side -->
        <geom name="studio_desk" type="box" pos="0.8 -1.3 0.35" size="0.4 0.6 0.35" rgba="0.45 0.32 0.22 1"/>
        <!-- RED/ORANGE floor mat indicates DEFAULT FALLBACK ROOM -->
        <geom name="creator_zone" type="box" pos="1.2 0 0.002" size="0.8 1.0 0.001" rgba="0.9 0.3 0.1 0.3" contype="0" conaffinity="0"/>
    </worldbody>
</mujoco>"""
    print("=" * 60 + "\n", flush=True)

    with open(output_xml_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(xml_output)
    return xml_output


if __name__ == "__main__":
    generate_room_xml()
