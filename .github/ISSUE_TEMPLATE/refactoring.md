---
name: Refactoring
about: 코드 구조 개선이나 리팩토링 시 사용
title: "♻️ [Refactor] "
labels: refactor
assignees: ''

---

## 🛠 리팩토링 대상
리팩토링하려는 파일 또는 모듈을 명시해주세요.

예) `UserService.java`
---

## 💡 개선 이유
리팩토링이 필요한 이유나 현재 문제점이 무엇인지 설명해주세요.

예)
- 단일 함수가 너무 많은 책임을 가짐 (SRP 위반)
- 코드 중복 발생
- 가독성이 떨어짐
- 테스트 작성이 어려움

---

## ✨ 개선 방향
리팩토링을 통해 달성하고자 하는 목표나 개선 방안을 구체적으로 작성해주세요.

예)
- `UserService`를 역할 단위로 분리 (등록/조회/삭제)
- 공통 로직을 `utils/validation.js`로 이동
- React custom hook 사용으로 상태관리 일원화

---

## ✅ 체크리스트
- [ ] 기존 기능에 영향 없음
- [ ] 테스트 코드 수정 또는 추가
- [ ] 중복 코드 제거
- [ ] 네이밍 컨벤션 준수

---

## 📎 참고 사항
관련 문서, 기준, 리뷰 포인트 등

예) [Clean Code 원칙](https://example.com/clean-code-guideline)
