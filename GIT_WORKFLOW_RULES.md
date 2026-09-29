# Git Workflow (Bản Ngắn Gọn)

## 1. Nhánh Chính
- **`main`**: Production (Không commit trực tiếp).
- **`develop`**: Tích hợp code (Không commit trực tiếp).

## 2. Đặt Tên Nhánh
Cú pháp: `<loại>/<tên-nhánh>`
- `feature/...`: Tính năng mới (VD: `feature/login`)
- `bugfix/...`: Sửa lỗi (VD: `bugfix/ui-error`)
- `hotfix/...`: Lỗi gấp production, tách từ `main` (VD: `hotfix/crash`)

## 3. Commit Message
Cú pháp: `<type>(scope): <mô tả>`
- `feat`: Thêm tính năng mới
- `fix`: Sửa lỗi
- `docs`: Tài liệu (README...)
- `refactor`: Tái cấu trúc code
- `chore`: Cấu hình, thư viện...
*Ví dụ: `feat(auth): thêm đăng nhập google`*

## 4. Luồng Code Thực Tế
1. **Lấy code mới nhất**: `git checkout develop` -> `git pull origin develop`
2. **Tạo nhánh**: `git checkout -b feature/tên-nhánh`
3. **Commit**: Cứ code xong là `git commit -m "feat: mô tả ngắn"`
4. **Chuẩn bị Push**: `git pull origin develop` (xử lý conflict nếu có)
5. **Push code**: `git push origin feature/tên-nhánh`
6. **Tạo PR/MR**: Lên Github tạo PR vào nhánh `develop`. **KHÔNG TỰ MERGE**.
7. **Review & Merge**: Nhờ đồng đội review -> **Squash and Merge** -> Xóa nhánh.
