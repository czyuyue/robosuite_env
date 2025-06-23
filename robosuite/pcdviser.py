import numpy as np
import json
import webbrowser
import tempfile
import os
import socket
from typing import Union, Optional, Tuple
import base64

class PointCloudVisualizer:
    def __init__(self):
        pass
    
    def _convert_to_numpy(self, pcd) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """将不同格式的点云数据转换为numpy数组"""
        points = None
        colors = None
        
        # 处理Open3D点云
        if hasattr(pcd, 'points') and hasattr(pcd, 'colors'):
            points = np.asarray(pcd.points)
            if len(pcd.colors) > 0:
                colors = np.asarray(pcd.colors)
        elif hasattr(pcd, 'points'):
            points = np.asarray(pcd.points)
            
        # 处理numpy数组
        elif isinstance(pcd, np.ndarray):
            if pcd.shape[1] >= 3:
                points = pcd[:, :3]
                if pcd.shape[1] >= 6:  # 假设后3列是颜色
                    colors = pcd[:, 3:6]
                    # 如果颜色值在0-255范围，归一化到0-1
                    if colors.max() > 1.0:
                        colors = colors / 255.0
        
        # 处理字典格式
        elif isinstance(pcd, dict):
            if 'points' in pcd:
                points = np.array(pcd['points'])
            if 'colors' in pcd:
                colors = np.array(pcd['colors'])
                if colors.max() > 1.0:
                    colors = colors / 255.0
        
        if points is None:
            raise ValueError("无法从输入数据中提取点云坐标")
            
        return points, colors
    
    def _prepare_data(self, points: np.ndarray, colors: Optional[np.ndarray] = None) -> dict:
        """准备用于Web可视化的数据"""
        # 下采样大点云以提高性能
        if len(points) > 50000:
            indices = np.random.choice(len(points), 50000, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
            print(f"点云过大，已下采样到50,000个点")
        
        # 居中和缩放点云
        center = np.mean(points, axis=0)
        points_centered = points - center
        scale = np.max(np.abs(points_centered))
        if scale == 0:
            scale = 1
        points_normalized = points_centered / scale
        
        data = {
            'points': points_normalized.tolist(),
            'center': center.tolist(),
            'scale': float(scale),
            'count': len(points)
        }
        
        if colors is not None:
            data['colors'] = colors.tolist()
        
        return data

    def _find_free_port(self, start_port: int = 8080) -> int:
        """找到可用端口"""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        raise RuntimeError("无法找到可用端口")

    def visualize_standalone_html(self, pcd, filename: str = None):
        """
        生成独立的HTML文件，不需要服务器
        """
        try:
            points, colors = self._convert_to_numpy(pcd)
            print(f"点云包含 {len(points)} 个点")
            
            data = self._prepare_data(points, colors)
            
            if filename is None:
                filename = tempfile.mktemp(suffix='.html')
            
            html_content = self._generate_standalone_html(data)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML文件已生成: {filename}")
            webbrowser.open(f'file://{os.path.abspath(filename)}')
            
            return filename
            
        except Exception as e:
            print(f"错误: {e}")
            return None

    def _generate_standalone_html(self, data: dict) -> str:
        """生成独立的HTML文件"""
        html_template = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>点云可视化</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: linear-gradient(45deg, #1a1a2e, #16213e);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
        }}
        #info {{
            position: absolute;
            top: 20px;
            left: 20px;
            color: white;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 10px;
            font-size: 14px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        #controls {{
            position: absolute;
            top: 20px;
            right: 20px;
            color: white;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 10px;
            font-size: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .info-item {{
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .info-label {{
            font-weight: bold;
            margin-right: 10px;
        }}
        .info-value {{
            color: #4CAF50;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
            text-align: center;
        }}
        .loading-spinner {{
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div id="loading">
        <div class="loading-spinner"></div>
        <div>加载点云数据中...</div>
    </div>
    <div id="container"></div>
    <div id="info" style="display: none;">
        <div class="info-item">
            <span class="info-label">点云数量:</span>
            <span class="info-value" id="point-count"></span>
        </div>
        <div class="info-item">
            <span class="info-label">原始中心:</span>
            <span class="info-value" id="center"></span>
        </div>
        <div class="info-item">
            <span class="info-label">缩放比例:</span>
            <span class="info-value" id="scale"></span>
        </div>
    </div>
    <div id="controls" style="display: none;">
        <div style="font-weight: bold; margin-bottom: 10px; color: #4CAF50;">控制说明</div>
        <div>🖱️ 左键: 旋转视角</div>
        <div>🖱️ 右键: 平移视角</div>
        <div>🎯 滚轮: 缩放远近</div>
        <div>🔄 双击: 重置视角</div>
        <div>⌨️ 空格: 暂停/继续自动旋转</div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const data = {json.dumps(data)};
        let scene, camera, renderer, points;
        let isMouseDown = false;
        let mouseButton = 0;
        let mouseX = 0, mouseY = 0;
        let targetRotationX = 0, targetRotationY = 0;
        let targetX = 0, targetY = 0, targetZ = 3;
        let autoRotate = true;
        let autoRotateSpeed = 0.005;
        
        function init() {{
            // 显示信息
            document.getElementById('point-count').textContent = data.count.toLocaleString();
            document.getElementById('center').textContent = `(${{data.center.map(x => x.toFixed(2)).join(', ')}})`;
            document.getElementById('scale').textContent = data.scale.toFixed(2);
            
            // 初始化场景
            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setClearColor(0x000000, 0);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // 创建点云
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(data.points.flat());
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            let material;
            if (data.colors) {{
                const colors = new Float32Array(data.colors.flat());
                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                material = new THREE.PointsMaterial({{ 
                    size: 0.02, 
                    vertexColors: true,
                    sizeAttenuation: true,
                    transparent: true,
                    opacity: 0.8
                }});
            }} else {{
                material = new THREE.PointsMaterial({{ 
                    color: 0x00ff88, 
                    size: 0.02,
                    sizeAttenuation: true,
                    transparent: true,
                    opacity: 0.8
                }});
            }}
            
            points = new THREE.Points(geometry, material);
            scene.add(points);
            
            // 添加坐标轴
            const axesHelper = new THREE.AxesHelper(0.5);
            scene.add(axesHelper);
            
            // 添加环境光
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            // 添加定向光
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // 设置相机位置
            camera.position.set(2, 2, 3);
            camera.lookAt(0, 0, 0);
            
            // 设置事件监听器
            setupEventListeners();
            
            // 隐藏加载界面
            document.getElementById('loading').style.display = 'none';
            document.getElementById('info').style.display = 'block';
            document.getElementById('controls').style.display = 'block';
            
            // 开始渲染
            animate();
        }}
        
        function setupEventListeners() {{
            renderer.domElement.addEventListener('mousedown', (event) => {{
                isMouseDown = true;
                mouseButton = event.button;
                mouseX = event.clientX;
                mouseY = event.clientY;
                autoRotate = false;
            }});
            
            document.addEventListener('mouseup', () => {{
                isMouseDown = false;
            }});
            
            document.addEventListener('mousemove', (event) => {{
                if (!isMouseDown) return;
                
                const deltaX = event.clientX - mouseX;
                const deltaY = event.clientY - mouseY;
                
                if (mouseButton === 0) {{ // 左键旋转
                    targetRotationY += deltaX * 0.01;
                    targetRotationX += deltaY * 0.01;
                }} else if (mouseButton === 2) {{ // 右键平移
                    targetX += deltaX * 0.002;
                    targetY -= deltaY * 0.002;
                }}
                
                mouseX = event.clientX;
                mouseY = event.clientY;
            }});
            
            renderer.domElement.addEventListener('wheel', (event) => {{
                event.preventDefault();
                targetZ += event.deltaY * 0.002;
                targetZ = Math.max(0.5, Math.min(20, targetZ));
            }});
            
            renderer.domElement.addEventListener('contextmenu', (event) => {{
                event.preventDefault();
            }});
            
            // 双击重置
            renderer.domElement.addEventListener('dblclick', () => {{
                targetRotationX = 0;
                targetRotationY = 0;
                targetX = 0;
                targetY = 0;
                targetZ = 3;
                autoRotate = true;
            }});
            
            // 空格键控制自动旋转
            document.addEventListener('keydown', (event) => {{
                if (event.code === 'Space') {{
                    event.preventDefault();
                    autoRotate = !autoRotate;
                }}
            }});
            
            // 窗口大小变化处理
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            
            // 自动旋转
            if (autoRotate) {{
                targetRotationY += autoRotateSpeed;
            }}
            
            // 平滑相机移动
            camera.position.x += (targetX - camera.position.x) * 0.1;
            camera.position.y += (targetY - camera.position.y) * 0.1;
            camera.position.z += (targetZ - camera.position.z) * 0.1;
            
            // 旋转点云
            points.rotation.x += (targetRotationX - points.rotation.x) * 0.1;
            points.rotation.y += (targetRotationY - points.rotation.y) * 0.1;
            
            renderer.render(scene, camera);
        }}
        
        // 页面加载完成后初始化
        window.addEventListener('load', init);
    </script>
</body>
</html>'''
        return html_template

    def visualize_with_plotly(self, pcd):
        """使用Plotly进行可视化（需要安装plotly）"""
        try:
            import plotly.graph_objects as go
            import plotly.offline as offline
            
            points, colors = self._convert_to_numpy(pcd)
            print(f"点云包含 {len(points)} 个点")
            
            # 下采样
            if len(points) > 10000:
                indices = np.random.choice(len(points), 10000, replace=False)
                points = points[indices]
                if colors is not None:
                    colors = colors[indices]
                print(f"Plotly可视化：已下采样到10,000个点")
            
            # 创建scatter3d
            if colors is not None:
                # 将RGB转换为颜色字符串
                color_strings = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                               for r, g, b in colors]
                scatter = go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=color_strings,
                        opacity=0.8
                    ),
                    name='点云'
                )
            else:
                scatter = go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='lightgreen',
                        opacity=0.8
                    ),
                    name='点云'
                )
            
            fig = go.Figure(data=[scatter])
            fig.update_layout(
                title='点云可视化 (Plotly)',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='cube'
                ),
                width=1000,
                height=700
            )
            
            offline.plot(fig, auto_open=True)
            
        except ImportError:
            print("请安装plotly: pip install plotly")
        except Exception as e:
            print(f"Plotly可视化错误: {e}")

    def visualize_with_open3d(self, pcd):
        """使用Open3D进行可视化（需要安装open3d）"""
        try:
            import open3d as o3d
            
            points, colors = self._convert_to_numpy(pcd)
            
            # 创建Open3D点云对象
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(points)
            
            if colors is not None:
                o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 显示
            o3d.visualization.draw_geometries([o3d_pcd])
            
        except ImportError:
            print("请安装open3d: pip install open3d")
        except Exception as e:
            print(f"Open3D可视化错误: {e}")

# 创建全局可视化器
_visualizer = PointCloudVisualizer()

def visualize_pointcloud(pcd, method: str = 'html', filename: str = None):
    """
    点云可视化函数
    
    参数:
    - pcd: 点云数据（Open3D点云、numpy数组或字典）
    - method: 可视化方法
      - 'html': 生成独立HTML文件（推荐，无端口问题）
      - 'plotly': 使用Plotly可视化（需要安装plotly）
      - 'open3d': 使用Open3D可视化（需要安装open3d）
    - filename: HTML文件名（仅html方法有效）
    
    返回:
    - HTML文件路径（html方法）或None
    """
    if method == 'html':
        return _visualizer.visualize_standalone_html(pcd, filename)
    elif method == 'plotly':
        _visualizer.visualize_with_plotly(pcd)
    elif method == 'open3d':
        _visualizer.visualize_with_open3d(pcd)
    else:
        print(f"不支持的方法: {method}")
        print("支持的方法: 'html', 'plotly', 'open3d'")

# 使用示例
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    
    # 创建一个螺旋点云
    t = np.linspace(0, 4*np.pi, 2000)
    x = np.cos(t) * (1 + 0.1*t)
    y = np.sin(t) * (1 + 0.1*t)
    z = 0.1 * t
    
    points = np.column_stack([x, y, z])
    
    # 添加彩虹色
    colors = np.column_stack([
        (np.sin(t) + 1) / 2,      # 红色
        (np.cos(t + np.pi/3) + 1) / 2,  # 绿色
        (np.sin(t + 2*np.pi/3) + 1) / 2  # 蓝色
    ])
    
    pointcloud_data = np.column_stack([points, colors])
    
    print("测试点云可视化:")
    # print("1. HTML方法（推荐）")
    # visualize_pointcloud(pointcloud_data, method='html')
    
    # 可选：测试其他方法
    print("2. Plotly方法")
    visualize_pointcloud(pointcloud_data, method='plotly')
    
    # print("3. Open3D方法")
    # visualize_pointcloud(pointcloud_data, method='open3d')