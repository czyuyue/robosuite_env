import numpy as np
import os
import json
import open3d as o3d
import argparse
from pathlib import Path
import http.server
import socketserver
import webbrowser
import threading

def load_pointcloud(file_path):
    """
    加载点云文件
    支持 .ply, .pcd, .xyz, .npy 格式
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension in ['.ply', '.pcd', '.xyz']:
        # 使用Open3D加载点云
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        # 如果有颜色，也加载颜色
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            colors = np.ones_like(points) * 0.5  # 默认灰色
    elif extension == '.npy':
        # 加载NumPy数组
        points = np.load(file_path)
        if points.shape[1] >= 6:  # 假设后三列是RGB颜色
            colors = points[:, 3:6]
            points = points[:, :3]
        else:
            colors = np.ones_like(points) * 0.5  # 默认灰色
    else:
        raise ValueError(f"不支持的文件格式: {extension}")

    return points, colors

def generate_html(points, colors, output_dir, html_filename="index.html", point_size=0.02):
    """
    生成包含点云可视化的HTML页面

    参数:
        points: 点云坐标数组 (N, 3)
        colors: 点云颜色数组 (N, 3)，值范围为0-1
        output_dir: 输出目录
        html_filename: HTML文件名称
        point_size: 点的大小
    """
    os.makedirs(output_dir, exist_ok=True)

    # 将点云数据转换为JSON格式
    vertices = []
    for i in range(len(points)):
        vertices.append({
            "x": float(points[i, 0]),
            "y": float(points[i, 1]),
            "z": float(points[i, 2]),
            "r": float(colors[i, 0]),
            "g": float(colors[i, 1]),
            "b": float(colors[i, 2])
        })

    # 将数据保存为JSON文件
    json_path = os.path.join(output_dir, "pointcloud_data.json")
    with open(json_path, "w") as f:
        json.dump(vertices, f)

    # HTML模板
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>点云可视化</title>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        canvas {{ display: block; }}
        #info {{
            position: absolute;
            top: 10px;
            width: 100%;
            text-align: center;
            color: white;
            font-family: Arial;
            font-size: 18px;
            font-weight: bold;
            text-shadow: 1px 1px 2px black;
        }}
        .control-panel {{
            position: absolute;
            top: 50px;
            right: 20px;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
            color: white;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.min.js"></script>
</head>
<body>
    <div id="info">点云可视化 - 使用鼠标拖动旋转，滚轮缩放</div>
    <div class="control-panel">
        <div>点数量: <span id="point-count">0</span></div>
        <div>帧率: <span id="fps">0</span> FPS</div>
    </div>

    <script>
        // 初始化Three.js场景
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111);

        // 添加环境光和平行光
        scene.add(new THREE.AmbientLight(0x404040));
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        // 设置相机
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;

        // 设置渲染器
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // 添加轨道控制器
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;

        // 添加坐标轴辅助
        const axesHelper = new THREE.AxesHelper(2);
        scene.add(axesHelper);

        // 圆形点的着色器
        const vertexShader = `
            attribute float size;
            varying vec3 vColor;
            void main() {{
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (300.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }}
        `;

        const fragmentShader = `
            varying vec3 vColor;
            void main() {{
                vec2 xy = gl_PointCoord.xy - vec2(0.5);
                float r = length(xy);
                if (r > 0.5) discard;
                gl_FragColor = vec4(vColor, 1.0);
            }}
        `;

        // 加载点云数据
        fetch('pointcloud_data.json')
            .then(response => response.json())
            .then(data => {{
                // 创建几何体
                const geometry = new THREE.BufferGeometry();
                const positions = [];
                const colors = [];
                const sizes = [];
                const pointSize = {point_size};

                // 填充点和颜色数据
                for (const point of data) {{
                    positions.push(point.x, point.y, point.z);
                    colors.push(point.r, point.g, point.b);
                    sizes.push(pointSize);
                }}

                // 更新点数量显示
                document.getElementById('point-count').textContent = data.length;

                // 设置属性
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
                geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

                // 创建点云着色器材质，使点变成圆形而不是方形
                const material = new THREE.ShaderMaterial({{
                    uniforms: {{

                    }},
                    vertexShader: vertexShader,
                    fragmentShader: fragmentShader,
                    vertexColors: true,
                    transparent: true
                }});

                // 创建点云对象
                const pointCloud = new THREE.Points(geometry, material);
                scene.add(pointCloud);

                // 计算点云边界并调整相机位置
                geometry.computeBoundingSphere();
                const {{ center, radius }} = geometry.boundingSphere;

                // 将相机位置设置为正确观察点云的位置
                camera.position.set(center.x, center.y, center.z + radius * 2);
                controls.target.copy(center);
                controls.update();
            }})
            .catch(error => console.error('加载点云数据失败:', error));

        // 处理窗口大小变化
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // FPS计算
        let frameCount = 0;
        let lastTime = performance.now();

        // 动画循环
        function animate() {{
            requestAnimationFrame(animate);

            // 更新控制器
            controls.update();

            // 渲染场景
            renderer.render(scene, camera);

            // 计算FPS
            frameCount++;
            const currentTime = performance.now();
            if (currentTime - lastTime >= 1000) {{
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = currentTime;
            }}
        }}

        animate();
    </script>
</body>
</html>
"""

    # 将HTML内容写入文件
    html_path = os.path.join(output_dir, html_filename)
    with open(html_path, "w") as f:
        f.write(html_content)

    return html_path, json_path

def downsample_pointcloud(points, colors, target_count):
    """
    对点云进行降采样，以减小文件大小
    """
    if len(points) <= target_count:
        return points, colors

    # 简单随机采样
    indices = np.random.choice(len(points), target_count, replace=False)
    return points[indices], colors[indices]

def start_http_server(directory, port=8000):
    """
    启动HTTP服务器来托管HTML和JSON文件
    """
    # 切换到指定目录
    os.chdir(directory)

    # 创建HTTP服务器
    handler = http.server.SimpleHTTPRequestHandler

    # 允许自动选择可用端口
    for attempt_port in range(port, port + 100):
        try:
            httpd = socketserver.TCPServer(("", attempt_port), handler)
            print(f"启动HTTP服务器在 http://localhost:{attempt_port}")
            print(f"按Ctrl+C停止服务器")

            # 在浏览器中打开页面
            webbrowser.open(f"http://localhost:{attempt_port}")

            # 运行服务器
            httpd.serve_forever()
            break
        except OSError as e:
            if e.errno == 98:  # 端口已被占用
                print(f"端口 {attempt_port} 已被占用，尝试下一个端口...")
            else:
                raise
    else:
        print(f"无法找到可用端口（尝试了 {port} 到 {port+99}）")

def visualize_ndarray(points_array, colors=None, output_dir="output", port=8000, max_points=100000, point_size=0.002):
    """
    直接可视化numpy数组形式的点云数据

    参数:
        points_array: numpy数组，形状为(N, 3)或(N, 6)
                      如果是(N, 6)，后3列视为RGB颜色值
        colors: 可选，点云颜色数组，形状为(N, 3)，值范围为0-1
                如果为None且points_array为(N, 3)，则使用默认灰色
        output_dir: 输出目录，默认为"output"
        port: HTTP服务器端口，默认为8000
        max_points: 最大点数，超过会进行降采样，默认为100000
        point_size: 点的大小，默认为0.02

    返回:
        html_path: 生成的HTML文件路径
        json_path: 生成的JSON文件路径
    """
    # 检查输入数据形状
    if not isinstance(points_array, np.ndarray):
        raise TypeError("points_array必须是numpy.ndarray类型")

    if points_array.ndim != 2:
        raise ValueError(f"points_array必须是二维数组，当前维度: {points_array.ndim}")

    # 处理包含颜色信息的点云数据
    if points_array.shape[1] >= 6:
        # 如果没有提供colors参数，使用数组中的颜色信息
        if colors is None:
            colors = points_array[:, 3:6]
        points = points_array[:, :3]
    else:
        points = points_array[:, :3]
        # 如果没有颜色信息，使用默认的灰色
        if colors is None:
            colors = np.ones_like(points) * 0.5  # 默认灰色

    # 确保颜色值在0-1范围内
    if colors.max() > 1.0:
        colors = colors / 255.0

    # 如果点数过多，进行降采样
    if len(points) > max_points:
        print(f"点云包含 {len(points)} 个点，超过最大限制 {max_points}，进行降采样...")
        points, colors = downsample_pointcloud(points, colors, max_points)

    print(f"处理 {len(points)} 个点...")

    # 生成HTML
    html_path, json_path = generate_html(points, colors, output_dir, point_size=point_size)

    print(f"HTML文件已生成: {html_path}")
    print(f"点云数据文件: {json_path}")

    # 启动HTTP服务器在新线程中
    current_dir = os.getcwd()
    server_dir = os.path.abspath(output_dir)

    print(f"在浏览器中启动可视化，URL: http://localhost:{port}")

    # 在新线程中启动服务器
    server_thread = threading.Thread(
        target=start_http_server,
        args=(server_dir, port),
        daemon=True
    )
    server_thread.start()

    # 返回文件路径，便于后续使用
    return html_path, json_path, server_thread

def main():
    parser = argparse.ArgumentParser(description="点云可视化工具")
    parser.add_argument("--input", "-i", type=str, help="输入点云文件路径 (.ply, .pcd, .xyz, .npy格式)")
    parser.add_argument("--output", "-o", type=str, default="output", help="输出目录")
    parser.add_argument("--max_points", "-m", type=int, default=100000, help="最大点数 (默认: 100000)")
    parser.add_argument("--port", "-p", type=int, default=8000, help="HTTP服务器端口 (默认: 8000)")
    parser.add_argument("--random", "-r", action="store_true", help="生成随机点云数据 (不需要输入文件)")
    parser.add_argument("--point_size", "-s", type=float, default=0.02, help="点的大小 (默认: 0.02)")

    args = parser.parse_args()

    # 点云数据
    if args.random or args.input is None:
        print("生成随机点云数据...")
        points = np.random.rand(10000, 3)
        colors = np.ones_like(points) * 0.5  # 默认灰色
    else:
        # 加载点云
        print(f"加载点云文件: {args.input}")
        points, colors = load_pointcloud(args.input)

    # 如果点数过多，进行降采样
    if len(points) > args.max_points:
        print(f"点云包含 {len(points)} 个点，超过最大限制 {args.max_points}，进行降采样...")
        points, colors = downsample_pointcloud(points, colors, args.max_points)

    print(f"处理 {len(points)} 个点...")

    # 生成HTML
    html_path, json_path = generate_html(points, colors, args.output, point_size=args.point_size)

    print(f"HTML文件已生成: {html_path}")
    print(f"点云数据文件: {json_path}")

    # 启动HTTP服务器在新线程中
    current_dir = os.getcwd()
    server_dir = os.path.abspath(args.output)

    print(f"在浏览器中启动可视化，URL: http://localhost:{args.port}")

    # 在新线程中启动服务器
    server_thread = threading.Thread(
        target=start_http_server,
        args=(server_dir, args.port),
        daemon=True
    )
    server_thread.start()

    try:
        # 保持主线程运行，直到用户按Ctrl+C
        server_thread.join()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        # 恢复工作目录
        os.chdir(current_dir)

# 使用示例
if __name__ == "__main__":
    main()

"""
# 示例代码：直接使用ndarray进行可视化
import numpy as np
from visualize_pointcloud import visualize_ndarray

# 创建随机点云数据
points = np.random.rand(5000, 3)  # 5000个随机点
colors = np.random.rand(5000, 3)  # 随机颜色

# 可视化点云
html_path, json_path, server_thread = visualize_ndarray(
    points_array=points,
    colors=colors,
    point_size=0.01  # 更小的点大小
)

try:
    # 保持脚本运行，直到用户按Ctrl+C
    server_thread.join()
except KeyboardInterrupt:
    print("服务器已停止")
"""