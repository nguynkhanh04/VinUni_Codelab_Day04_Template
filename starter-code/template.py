"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas + ReAct Loop.
"""

import json
import os
import re
import sys
from typing import Dict, Any, List, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Available Tools, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của Tập đoàn Vingroup, chuyên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng trong hệ sinh thái Vingroup (VinFast, Vinpearl, VinWonders, Vinhomes, Vinmec,...).

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ khách hàng VinFast, Vinpearl.
- Phong cách giao tiếp: Chuyên nghiệp, lịch sự, thân thiện, tận tâm và chính xác.

## 2. AVAILABLE TOOLS
1. `search_product_catalog(category: str, max_price: int)`: Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục ('xe_dien' hoặc 'du_lich') và mức giá tối đa (VNĐ).
2. `submit_support_ticket(customer_name: str, issue_description: str, priority: str)`: Tạo phiếu hỗ trợ/khiếu nại kỹ thuật của khách hàng với mức độ ưu tiên ('low', 'medium', 'high').

## 3. CORE RULES
1. KHÔNG BAO GIỜ bịa đặt dữ liệu về giá cả, thông số kỹ thuật sản phẩm hoặc chính sách bảo hành.
2. BẮT BUỘC phải gọi công cụ `search_product_catalog` để tra cứu thông tin thực tế khi khách hàng hỏi về giá, dòng xe, kỳ nghỉ hoặc danh mục sản phẩm.
3. BẮT BUỘC phải gọi công cụ `submit_support_ticket` khi khách hàng báo lỗi sự cố, khiếu nại chất lượng dịch vụ hoặc cần hỗ trợ kỹ thuật.
4. Xử lý trung thực khi không có dữ liệu: Nếu không tìm thấy sản phẩm phù hợp, thông báo rõ ràng "Rất tiếc, không tìm thấy sản phẩm phù hợp." thay vì suy đoán.
5. Trả lời chính xác và ngắn gọn các câu hỏi chính sách thường gặp (FAQ) theo quy chuẩn của Vingroup.

## 4. OPERATIONAL BOUNDARIES
- Chỉ phục vụ và trả lời các thông tin liên quan đến hệ sinh thái Vingroup.
- Từ chối lịch sự nếu khách hàng hỏi về các chủ đề ngoài phạm vi hoạt động của Vingroup.

## 5. OUTPUT CONTRACT
Tuân thủ chu trình ReAct (Reasoning + Acting):
- Thought: Phân tích yêu cầu của khách hàng, xác định thông tin cần thu thập.
- Action: Tên công cụ cần thực thi kèm tham số (JSON), hoặc None nếu trả lời trực tiếp.
- Observation: Kết quả nhận được sau khi chạy công cụ.
- Final Answer: Tổng hợp câu trả lời hoàn chỉnh, lịch sự gửi tới khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Mô phỏng phản hồi của mô hình khi không có Tool Calling.
        Mục tiêu: Thể hiện nguy cơ bịa đặt thông tin (hallucination) do không tra cứu cơ sở dữ liệu thực.
        """
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}. (Không tra cứu cơ sở dữ liệu thực tế)",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering, Intent Detection & ReAct Loop."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # TODO 3: Phân tích Intent & Trích xuất tham số từ user_input
    # -----------------------------------------------------------------------
    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """
        Phân tích ý định người dùng và trích xuất tham số:
        - needs_catalog: Kiểm tra có nhu cầu tìm kiếm, xem sản phẩm/dịch vụ hay không.
        - needs_ticket: Kiểm tra có nhu cầu báo lỗi, khiếu nại, hỗ trợ kỹ thuật hay không.
        - is_faq: Nếu không cần gọi tool, xác định là câu hỏi thông tin chung.
        """
        user_lower = user_input.lower()

        # 1. Phát hiện Intent Ticket
        ticket_triggers = [
            "lỗi", "hỏng", "sự cố", "khiếu nại", "phản hồi", "ẩm mốc",
            "gặp vấn đề", "không hoạt động", "ticket", "xử lý gấp", "nghiêm trọng"
        ]
        needs_ticket = any(trigger in user_lower for trigger in ticket_triggers)

        ticket_args = {}
        if needs_ticket:
            name_match = re.search(
                r"(?:tên tôi là|tôi tên là|tôi tên|khách hàng)\s*[:\s]*([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\bxe\b|\bphòng\b|\bvà\b|\bđang\b|$)",
                user_input, re.IGNORECASE
            )
            customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

            if any(w in user_lower for w in ["gấp", "nghiêm trọng", "khẩn cấp", "high"]):
                priority = "high"
            elif any(w in user_lower for w in ["thấp", "nhẹ", "low"]):
                priority = "low"
            else:
                priority = "medium"

            issue_match = re.search(
                r"((?:xe|phòng)?[^,.]*?(?:lỗi|hỏng|ẩm mốc|sự cố)[^,.]*)",
                user_input, re.IGNORECASE
            )
            issue_description = issue_match.group(1).strip() if issue_match else user_input.strip()

            ticket_args = {
                "customer_name": customer_name,
                "issue_description": issue_description,
                "priority": priority
            }

        # 2. Phát hiện Intent Catalog
        is_faq_query = "bảo hành" in user_lower and not any(w in user_lower for w in ["dưới", "triệu", "tỷ", "mua", "xem", "giá"])

        catalog_triggers = ["xem", "tìm", "mua", "tham khảo", "báo giá", "giá", "dưới", "có xe", "resort"]
        has_catalog_intent = any(trigger in user_lower for trigger in catalog_triggers)
        needs_catalog = has_catalog_intent and not is_faq_query

        catalog_args = {}
        if needs_catalog:
            if any(w in user_lower for w in ["resort", "du lịch", "vinpearl", "khách sạn", "phòng"]):
                category = "du_lich"
            else:
                category = "xe_dien"

            max_price = 999999999999
            price_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(triệu|tỷ|tr|k)?", user_input, re.IGNORECASE)
            if price_match:
                val = float(price_match.group(1).replace(",", "."))
                unit = (price_match.group(2) or "").lower()
                if unit in ["tỷ", "ty"]:
                    max_price = int(val * 1_000_000_000)
                elif unit in ["triệu", "tr"]:
                    max_price = int(val * 1_000_000)
                elif unit == "k":
                    max_price = int(val * 1_000)
                elif val >= 100_000:
                    max_price = int(val)

            catalog_args = {
                "category": category,
                "max_price": max_price
            }

        is_faq = not needs_catalog and not needs_ticket

        return {
            "needs_catalog": needs_catalog,
            "catalog_args": catalog_args,
            "needs_ticket": needs_ticket,
            "ticket_args": ticket_args,
            "is_faq": is_faq
        }

    # -----------------------------------------------------------------------
    # Trả lời câu hỏi FAQ từ cơ sở tri thức
    # -----------------------------------------------------------------------
    def _answer_faq(self, user_input: str) -> str:
        """Trả lời các câu hỏi thường gặp (FAQ) về chính sách của Vingroup."""
        faq_kb = {
            "bảo hành": (
                "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm (hoặc không giới hạn số km tuỳ chính sách pin). "
                "Bên cạnh đó, xe điện VinFast cũng được áp dụng chế độ bảo hành chính hãng lên đến 10 năm hoặc 200.000 km."
            )
        }
        for kw, ans in faq_kb.items():
            if kw in user_input.lower():
                return ans
        return "Thông tin chính sách đang được cập nhật. Quý khách vui lòng liên hệ tổng đài để được giải đáp chi tiết."

    # -----------------------------------------------------------------------
    # Tổng hợp Final Answer từ các observations trong trace
    # -----------------------------------------------------------------------
    def _synthesize_answer(self, user_input: str) -> str:
        """Tổng hợp câu trả lời cuối cùng từ các quan sát (observations) trong trace."""
        sections = []

        for step in self.trace:
            action = step.get("action")
            if not action:
                continue

            tool_name = action.get("tool")
            obs = step.get("observation")

            if tool_name == "search_product_catalog":
                if not obs:
                    sections.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    lines = ["Danh sách sản phẩm phù hợp tìm thấy:"]
                    for idx, item in enumerate(obs, 1):
                        name = item.get("name", "Sản phẩm")
                        price = item.get("price_vnd", 0)
                        desc = item.get("description", "")
                        lines.append(f"{idx}. {name} — Giá: {price:,.0f} VNĐ. {desc}")
                    sections.append("\n".join(lines))

            elif tool_name == "submit_support_ticket":
                if isinstance(obs, dict):
                    ticket_id = obs.get("ticket_id", "")
                    cust_name = obs.get("customer_name", "Quý khách")
                    priority = obs.get("priority", "medium")
                    msg = obs.get("message", f"Ticket {ticket_id} đã được tạo thành công.")
                    sections.append(
                        f"Yêu cầu hỗ trợ của quý khách {cust_name} đã được tiếp nhận.\n"
                        f"- Mã phiếu (Ticket ID): {ticket_id}\n"
                        f"- Mức độ ưu tiên: {priority}\n"
                        f"- Thông báo: {msg}"
                    )

        if not sections:
            return "Không có thông tin để hiển thị."

        return "\n\n".join(sections)

    # -----------------------------------------------------------------------
    # TODO 4: Xây dựng Agent Loop & ReAct Step Execution
    # -----------------------------------------------------------------------
    def _execute_step(self, user_input: str, intents: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Thực hiện một bước trong ReAct Agent Loop:
        Thought -> Action -> Observation.
        Trả về: (result, is_final)
        """
        executed_tools = [step["action"]["tool"] for step in self.trace if step.get("action")]

        # 1. Câu hỏi FAQ: Trả lời trực tiếp không cần gọi tool
        if intents["is_faq"]:
            thought = "Câu hỏi thuộc chính sách / FAQ, trả lời trực tiếp mà không cần gọi tool."
            answer = self._answer_faq(user_input)
            self.trace.append({
                "thought": thought,
                "action": None,
                "observation": None,
                "final_answer": answer
            })
            return answer, True

        # 2. Cần gọi search_product_catalog
        if intents["needs_catalog"] and "search_product_catalog" not in executed_tools:
            args = intents["catalog_args"]
            thought = f"Khách hàng cần tra cứu sản phẩm. Gọi search_product_catalog với tham số: {args}."
            observation = search_product_catalog(**args)

            step = {
                "thought": thought,
                "action": {"tool": "search_product_catalog", "arguments": args},
                "observation": observation
            }
            self.trace.append(step)

            # Nếu không cần gọi thêm ticket, hoàn tất và xuất Final Answer ngay
            if not intents["needs_ticket"]:
                answer = self._synthesize_answer(user_input)
                step["final_answer"] = answer
                return answer, True

            return "", False

        # 3. Cần gọi submit_support_ticket
        if intents["needs_ticket"] and "submit_support_ticket" not in executed_tools:
            args = intents["ticket_args"]
            thought = f"Khách hàng cần tạo phiếu hỗ trợ. Gọi submit_support_ticket với tham số: {args}."
            observation = submit_support_ticket(**args)

            step = {
                "thought": thought,
                "action": {"tool": "submit_support_ticket", "arguments": args},
                "observation": observation
            }
            self.trace.append(step)

            # Nếu không cần gọi catalog nữa (hoặc đã gọi trước đó)
            if not intents["needs_catalog"]:
                answer = self._synthesize_answer(user_input)
                step["final_answer"] = answer
                return answer, True

            return "", False

        # 4. Đã gọi đủ các tool cần thiết -> Tổng hợp Final Answer
        thought = "Đã thu thập đầy đủ quan sát từ các công cụ, tiến hành tổng hợp câu trả lời cuối cùng."
        answer = self._synthesize_answer(user_input)
        self.trace.append({
            "thought": thought,
            "action": None,
            "observation": None,
            "final_answer": answer
        })
        return answer, True

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop có kiểm soát Safeguard (Max Iterations)."""
        self.trace = []
        intents = self._detect_intents(user_input)

        iteration = 1
        while iteration <= self.max_iterations:
            result, is_final = self._execute_step(user_input, intents)
            if is_final:
                return {
                    "answer": result,
                    "trace": self.trace,
                    "status": "completed",
                    "iterations": iteration
                }
            iteration += 1

        # Vượt quá số bước tối đa (Safeguard)
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "status": "max_iterations_reached",
            "iterations": iteration - 1
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:\n", result["answer"])
    print("\nTrace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
